import asyncio
import base64
import json
import logging
import re
import shutil
from datetime import datetime
from openai import AsyncOpenAI
import aio_pika
from funasr import AutoModel
from pyannote.audio import Pipeline
import torch
import time
import zhconv
import librosa
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import threading

logging.basicConfig(
    level=logging.WARNING, # 生产环境设为 WARNING
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ivr_bot_error.log"), # 仅记录警告和错误
        logging.StreamHandler() # 控制台依然可以看到
    ]
)
logger = logging.getLogger("IVRBot_VLLM")

# 创建一个用于存放临时音档的目录
# sudo mkdir -p /mnt/ivr_ramdisk
# 将其挂载为 tmpfs（内存文件系统），分配 2GB 空间
# sudo mount -t tmpfs -o size=2G tmpfs /mnt/ivr_ramdisk
# 给权限，确保你的 Python 程序可以读写
# sudo chmod 777 /mnt/ivr_ramdisk

# 🔹 全局并发控制（根据显存压测调整）
MAX_CONCURRENT = 100
MQ_MAX_CONCURRENT = 100
semaphore = asyncio.Semaphore(MAX_CONCURRENT)
TEMP_DIR = "tmp/"
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
RAM_DISK_DIR = "/mnt/ivr_ramdisk/"
# SOURCE_DIR = "/opt/ezvoicetek/ezivr7000/userdata/recording/"
MQ_SERVER = "amqp://consilium:root@118.163.170.37/"
MQ_QUEUE = "genAI.filestream"
MQ_EXCHANGE = "genAI.exchange"
MQ_ROUTE = "genAI.filetrans"
MQ_REPLY = "genAI.response.v2"
MQ_REPLY_ROUTE = "genAI.response.v2"

def _clean_tags(text):
    """清洗 ASR 輸出的標籤與冗餘符號"""
    if not text: return ""
    # 移除 <|...|> 標籤
    text = re.sub(r'<\|.*?\|>|<>', '', text)
    # 移除情緒標籤如 [EMO] 等 (如果需要)
    # text = re.sub(r'\[.*?\]', '', text)
    return text.strip()


class AsyncAIBot:
    def __init__(self, vllm_base_url: str, api_key: str = "dummy"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=VLLM_BASE_URL)
        self.model_name = "qwen-collection"
        self._counts = defaultdict(int)
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=180)
        # 实际生产中建议从配置文件/DB 加载 Prompt
        self.system_prompt = "你是语音智能机器人，自动应答客户对话。请极简回覆，需要快速响应。"
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=False,
            # remote_code="./model.py",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            disable_update=True,
            device="cuda"  # 使用 GPU
        )
        self.correction_map = {
            "去讲": "去缴",
            "没法讲": "没法缴",
            "无法讲": "无法缴",
            "没办法讲": "没办法缴",
            "没法去讲": "没法去缴",
            "没法交": "没法缴"  # 预防交/缴混用
        }
        # Pyannote 说话人聚类模型
        self.hf_token = "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"  # TODO: 替换为您的 Hugging Face Token
        try:
            print("正在初始化说话人聚类引擎 (Pyannote)...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=True
                # use_auth_token=self.hf_token
            )
            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
        except Exception as e:
            print(f"Pyannote 初始化失败: {e}")
            self.diarization_pipeline = None

        self.common_rules = self._load_prompt("common_rules.json")
        self.ivr_logic = self._load_prompt("edc01.json")
        self.kaiji_collection = self._load_prompt("kaiji_collection.json")
        self.simulate_logic = self._load_prompt("simulate.json")

        self.collection_system_prompt = f"""你是凱基銀行專業催收專員。
                分析對話中的還款意願和風險點。
                {self.kaiji_collection}
                """
        # IVRBot
        self.ivr_system_prompt = f"""你是「東森傳家寶會員服務中心」的 AI 語音專員，負責處理會員對「K-SPARK 高雄演唱會」活動通知的即時回應。
                {self.ivr_logic}
                """
        # 物流場景 Prompt
        self.logistics_system_prompt = f"""你是專業物流服務質檢員。
                任務：分析物流取件、信息確認、包裹狀態等溝通情況。
                重點觀察：座席是否準確核對地址、單號，是否清晰說明領取時間（12點前）和標籤粘貼規則。
                {self.common_rules}
                """

        # 通用對話分析 Prompt
        self.qa_analysis_system_prompt = f"""你是一個專業的對話復盤助手。
                任務：根據提供的轉寫文本，清理出邏輯清晰的對話流，並提取核心信息。
                {self.common_rules}
                """

        # 模擬測試 Prompt
        self.simulate_system_prompt = f"""你是語音智能機器人，自動應答客戶對話。請極簡回覆。
                {self.simulate_logic}
                """

    def _load_prompt(self, filename):
        """从 configs 目录加载 Prompt 配置文件"""
        # 获取当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "configs", filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("prompt", "")
        except Exception as e:
            print(f"加载配置文件 {filename} 失败: {e}")
            return ""

    def _apply_business_logic(self, text):
        """基于词表进行硬替换"""
        corrected_text = text
        for wrong, right in self.correction_map.items():
            if wrong in corrected_text:
                # 使用 replace 进行全局替换
                corrected_text = corrected_text.replace(wrong, right)

        # 如果发生了修正，可以在这里记录日志便于后续分析模型表现
        if corrected_text != text:
            print(f"修正前: {text} -> 修正后: {corrected_text}")

        return corrected_text

    def increment(self, session_id: str, amount: int = 1):
        """安全地對 sessionId 進行加計"""
        with self._lock:
            self._counts[session_id] += amount

    def get_count(self, session_id: str) -> int:
        """安全地獲取計數"""
        with self._lock:
            return self._counts[session_id]

    def clear_session(self, session_id: str):
        """✅ 安全地清空並移除某個 sessionId，釋放記憶體"""
        with self._lock:
            # 使用 pop 帶有預設值，防止已被其他執行緒刪除時噴 KeyError
            self._counts.pop(session_id, None)

    async def process_chat_text(self, text: str, scenario: str, session_id: str = None, is_end: bool = False):
        if scenario == 'simulate':
            current_system_prompt = self.simulate_system_prompt
        else:
            current_system_prompt = self.ivr_system_prompt

        self.increment(session_id=session_id)
        
        response = await self.client.chat.completions.create(
            model="qwen-collection",
            messages=[
                    {'role': 'system', 'content': current_system_prompt},
                    {'role': 'user', 'content': f"客户说的话：{text}"},
                ],
            temperature=0.7,
            max_tokens=256
        )
        reply = zhconv.convert(response.choices[0].message.content, 'zh-tw')

        # 兼容 JSON 或纯文本回复
        match = re.search(r'\{[\s\S]*\}', reply)
        analysis = json.loads(match.group()) if match else {"reply": reply}
        print(f"决策结果: \n{analysis}")
        return text, analysis
    async def listen_and_analyze(self, audio_path, scenario):
        """
        输入：本地音频文件路径
        处理：Pyannote 说话人聚类 + SenseVoice ASR + LLM 意图分析
        """
        print(f"\n[1/3] 正在处理录音: {audio_path}")
        start_time = time.time()

        combined_text = ""

        # 1. 说话人聚类 (Diarization)
        try:
            # IVRBot 场景下，通常只有客户一方说话，或者为了追求极致速度（2s），直接跳过聚类
            if self.diarization_pipeline and scenario != 'IVRBot':
                print("[*] 正在执行说话人聚类 (Diarization)...")
                diarization = self.diarization_pipeline(audio_path)

                # 加载并切片
                audio_data, sr = librosa.load(audio_path, sr=16000)
                segments_results = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_sample = int(turn.start * sr)
                    end_sample = int(turn.end * sr)
                    chunk = audio_data[start_sample:end_sample]

                    if len(chunk) < 1600:
                        continue
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(
                        self.executor,
                        lambda: self.asr_model.generate(input=chunk, cache={}, language="auto", use_itn=True,
                                                  hotword="好喔, 考慮, 沒興趣", vad_kwargs={"max_end_silence_time": 800})
                    )

                    raw_item_text = res[0]['text']
                    clean_text = self._apply_business_logic(_clean_tags(raw_item_text))
                    if clean_text:
                        segments_results.append(f"[{speaker}]: {clean_text}")

                combined_text = "\n".join(segments_results)
            else:
                # 极速路径：直接使用 AutoModel 生成
                print("[*] 正在快速执行语音识别 (ASR)...")
                loop1 = asyncio.get_event_loop()
                res = await loop1.run_in_executor(
                    self.executor,
                    lambda: self.asr_model.generate(
                        input=audio_path,
                        cache={},
                        language="auto",
                        use_itn=True,
                        hotword="好喔, 考慮, 沒興趣",
                        vad_kwargs={"max_end_silence_time": 800}
                    )
                )
                combined_text = _clean_tags(res[0]['text'])
                combined_text = self._apply_business_logic(combined_text)
        except Exception as e:
            print(f"音檔處理失敗: {e}")
            return None, "音檔問題，無法識別"

        print(f"识别结果 (原始): \n{combined_text}")

        # 1.5 角色自动映射
        text_for_llm = combined_text
        # IVRBot 不需要复杂的角色映射，直接分析文本即可
        if self.diarization_pipeline and combined_text and scenario != 'IVRBot':
            print("[*] 正在通过 LLM 进行角色自动映射...")
            try:
                role_resp = self.client.chat.completions.create(model='Qwen2.5-7B-Instruct-AWQ', messages=[
                    {'role': 'system',
                     'content': "你是「東森傳家寶會員服務中心」的 AI 語音專員，負責處理會員對「K-SPARK 高雄演唱會」活動通知的即時回應。"},
                    {'role': 'user', 'content': f"/no_think\n {combined_text}"}
                ],
                temperature=0.6,
                max_tokens=1024,  # 限制生成长度
                )

                content = role_resp.choices[0].message.content
                role_json_str = re.search(r'\{.*\}', content, re.DOTALL)
                if role_json_str:
                    role_map = json.loads(role_json_str.group())
                    for spk_id, role_name in role_map.items():
                        text_for_llm = text_for_llm.replace(f"[{spk_id}]", f"[{role_name}]")
                    # print(f"映射结果: {role_map}", encoding=False)
            except Exception as e:
                print(f"角色映射失败: {e}")

        print(f"最终待分析文本: \n{text_for_llm}")
        print(f"感知耗时: {time.time() - start_time:.2f}s")

        # --- 空文本拦截：防止静音时 LLM 产生非 JSON 幻觉 ---
        if not text_for_llm.strip():
            logger.warning("识别结果为空，跳过 LLM 推理，直接返回默认回应")
            silent_response = {
                "場景類型": "無回應",
                "意圖": "沉默",
                "風險級別": "低",
                "策略": "重撥或待定",
                "回復客戶": "..."
            }
            return text_for_llm, silent_response

        # 2. 意图与逻辑分析 (DeepSeek)
        print("[2/3] 正在通过 DeepSeek 分析意图...")
        llme_start = time.time()

        text_for_llm = combined_text
        try:
            # 根据场景选择 System Prompt
            current_system_prompt = ""
            if scenario == 'collection':
                # current_system_prompt = self.collection_system_prompt
                current_system_prompt = self.ivr_system_prompt
            elif scenario == 'logistics':
                current_system_prompt = self.logistics_system_prompt
            elif scenario == 'IVRBot':
                current_system_prompt = self.ivr_system_prompt
            elif scenario == 'simulate':
                current_system_prompt = self.simulate_system_prompt
            else:
                current_system_prompt = self.qa_analysis_system_prompt

            response = await self.client.chat.completions.create(
                model="qwen-collection",  # 必须与 vLLM 启动时的 --served-model-name 一致
                messages=[
                    {'role': 'system', 'content': current_system_prompt},
                    {'role': 'user', 'content': f"客户说的话：{text_for_llm}"},
                ],
                temperature=0.0,  # 采样温度
                max_tokens=150,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            analysis = response.choices[0].message.content
            match = re.search(r'\{[\s\S]*\}', analysis)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                # print(json.dumps(data, ensure_ascii=False))
                analysis = data
            else:
                print("未找到 JSON 数据")
            print(f"决策结果: \n{analysis}")
            print(f"大脑决策耗时: {time.time() - llme_start:.2f}s")
            return text_for_llm, analysis
        except Exception as e:
            print(f"LLM 模型未启动或报错，请确保运行了。vLLM API Server错误: {e}")
            return text_for_llm, "LLM 分析失敗"

    async def run_simulated_call(self, audiofile, scenario):
        demo_audio = audiofile
        sst_result = ""
        resp = ""
        if os.path.exists(demo_audio):
            sst_result, resp = await self.listen_and_analyze(demo_audio, scenario)
            # if scenario == 'IVRBot':
            #     data = json.loads(resp)
            #     need_speak = data['回復客戶']
            #     need_speak = zhconv.convert(need_speak, 'zh-cn')
            #     wav_file = generate_wav_filename()
            #     data['wav_file'] = wav_file
            #     resp = json.dumps(data)
            #     asyncio.run(generate_wav(need_speak, wav_file))

        else:
            print("目前模型已加载到 GPU，等待输入流...")
        return sst_result, resp

    def reload_simulate_prompt(self):
        """重新加载模拟测试的 Prompt"""
        self.simulate_logic = self._load_prompt("simulate.json")
        self.simulate_system_prompt = f"""你是語音智能機器人，自動應答客戶對話。請極簡回覆。
        {self.simulate_logic}
        """
        print("已成功重新加载 simulate.json 配置文件。")



class RabbitMQConsumer:
    def __init__(self, mq_url: str, queue: str, exchange: str, routing_key: str):
        self.queue = None
        self.exchange = None
        self.mq_url = mq_url
        self.queue_name = queue
        self.exchange_name = exchange
        self.routing_key = routing_key
        self.bot = AsyncAIBot(vllm_base_url=VLLM_BASE_URL, api_key="Consilium")
        self.conn = None
        self.channel = None

    async def connect(self):
        self.conn = await aio_pika.connect_robust(self.mq_url)
        self.channel = await self.conn.channel()
        # QoS 必须 >= Python侧并发数，否则消息会堆积在 Broker 侧
        await self.channel.set_qos(prefetch_count=MQ_MAX_CONCURRENT)

        self.exchange = await self.channel.declare_exchange(
            self.exchange_name, aio_pika.ExchangeType.DIRECT, durable=True
        )
        self.queue = await self.channel.declare_queue(self.queue_name, durable=True)
        await self.queue.bind(self.exchange, self.routing_key)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RabbitMQ connected successfully")

    async def process_message(self, message: aio_pika.IncomingMessage):
        """单条消息处理流（自动 ACK / NACK）"""
        async with semaphore:  # 🔹 限流入口
            async with message.process():  # 成功自动 ACK，抛异常自动 NACK
                try:
                    body = json.loads(message.body.decode("utf-8"))
                    # print(body)
                    langChoose = body.get('lang')
                    createTime = body.get("createTime")
                    scenario = body.get('scenario')
                    if langChoose == 'ClearCache':
                        requestId = body.get("ucid")
                        self.bot.clear_session(requestId)
                        print(f"ucid {requestId}已清除")
                        return

                    if not scenario:
                        scenario = langChoose
                    if langChoose == "IVRTxtBot":
                        requestId = body.get("ucid")
                        snd2Bot = body.get("snd2Bot")

                    elif langChoose == 'simulate':
                        requestId = body.get('requestId')
                        audioFile = body.get('audioFile')
                        session_id = requestId
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        print(f"收到模擬測試需求 (Scenario: {scenario})，正在準備 configs/simulate.json...")
                        try:
                            if scenario == 'edc01':
                                with open(os.path.join(base_dir, "configs", "edc01.md"), 'r', encoding='utf-8') as md_f:
                                    md_content = md_f.read()
                                with open(os.path.join(base_dir, "configs", "simulate.json"), 'w', encoding='utf-8') as js_f:
                                    json.dump({"prompt": md_content}, js_f, ensure_ascii=False, indent=4)
                            else:
                                shutil.copy(os.path.join(base_dir, "configs", "kaiji_collection.json"),
                                            os.path.join(base_dir, "configs", "simulate.json"))

                            config_path = os.path.join(base_dir, "configs", "simulate.json")
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)

                            old_prompt = config_data.get("prompt", "")
                            new_logic_section = audioFile
                            pattern = r'【判定邏輯矩陣】.*?(?=\s+必須返回以下標準 JSON 格式)'
                            if re.search(pattern, old_prompt, re.DOTALL):
                                new_prompt = re.sub(pattern, new_logic_section, old_prompt, flags=re.DOTALL)
                                config_data["prompt"] = new_prompt
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                                self.bot.reload_simulate_prompt()
                            else:
                                print("未能在 simulate.json 中找到 【判定邏輯矩陣】 標記")
                            print("成功更新模擬配置")
                        except Exception as e:
                            print(f"更新模擬配置失敗: {e}")
                    elif langChoose == 'text_chat':
                        requestId = body.get('requestId')
                        audioFile = body.get('audioFile')
                        session_id = requestId
                        chat_text = audioFile
                        is_end = False
                        if not is_end:
                            clean_text, resp_data = await self.bot.process_chat_text(
                                text=chat_text, scenario=scenario,
                                session_id=session_id, is_end=is_end
                            )
                            print(clean_text)
                        else:
                            
                            resp_data = await self.bot.generate_summary(session_id=session_id)
                            self.bot.sessions.pop(session_id, None)  # 释放内存
                        
                    elif langChoose == 'deploy':
                        target_file = "kaiji_collection.json"
                        if scenario == 'logistics':
                            target_file = "common_rules.json"
                        elif scenario == 'edc01':
                            target_file = "edc01.json"

                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        print(f"收到正式部署需求 (Target: {target_file})，正在更新...")
                        try:
                            config_path = os.path.join(base_dir, "configs", target_file)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_base = os.path.splitext(target_file)[0]
                            backup_ext = os.path.splitext(target_file)[1]
                            backup_path = os.path.join(base_dir, "configs", f"{file_base}_{timestamp}{backup_ext}")
                            shutil.copy(config_path, backup_path)
                            print(f"備份成功: {backup_path}")
                            with open(config_path, 'r', encoding='utf-8') as f:
                                if target_file.endswith('.md'):
                                    old_prompt = f.read()
                                else:
                                    config_data = json.load(f)
                                    old_prompt = config_data.get("prompt", "")
                            simulate_path = os.path.join(base_dir, "configs", "simulate.json")
                            with open(simulate_path, 'r', encoding='utf-8') as f_1:
                                config_data = json.load(f_1)
                                new_logic_section = config_data.get("prompt", "")
                            new_prompt = new_logic_section
                            if target_file.endswith('.md'):
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    f.write(new_prompt)
                            else:
                                config_data["prompt"] = new_prompt
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                                        
                                if target_file == "kaiji_collection.json":
                                    self.bot.reload_kaiji_prompt()
                                elif target_file == "edc01.json":
                                    self.bot.reload_edc01_prompt()
                                else:
                                    self.bot.reload_common_rules_prompt()

                        except Exception as e:
                            print(f"部署配置更新失敗: {e}")

                        # ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                    else:
                        requestId = body.get("requestId")
                        audioFile = body.get("audioFile")
                        langChoose = body.get("lang")
                        sDate = body.get("date")
                        audioBytes = body.get("audioBytes")
                        # scenario = body.get('scenario')
                        # 音频字节流转wav
                        WAV_PATH = RAM_DISK_DIR + audioFile
                        with open(WAV_PATH, "wb") as f:
                            f.write(base64.b64decode(audioBytes))
                        if langChoose == 'simulate':
                            audioFile = f"temp_simulate_{requestId}.wav"

                    # source_path = os.path.join(SOURCE_DIR, audioFile)
                    # ram_path = os.path.join(RAM_DISK_DIR, audioFile)
                    # if os.path.exists(source_path):
                    #     shutil.copy(source_path, ram_path)
                    # else:
                    #     logger.error(f"File not found: {source_path}")
                    #     return
                    clean_text = ""
                    resp_data = {}
                    resp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        if langChoose == 'IVRTxtBot':
                            chat_text = "" if snd2Bot is None else snd2Bot

                            clean_text, resp_data = await self.bot.process_chat_text(
                                text=chat_text, scenario="IVRTxtBot", session_id=requestId)
                        elif langChoose == 'text_chat':
                            chat_text = audioFile
                            clean_text, resp_data = await self.bot.process_chat_text(
                                text=chat_text, scenario=scenario,
                                session_id=session_id
                            )
                        #     else:
                        #         resp_data = await self.bot.generate_summary(session_id=session_id)
                        #         self.bot.sessions.pop(session_id, None)  # 释放内存
                        else:
                            sst_result, resp_text = await self.bot.run_simulated_call(audiofile=WAV_PATH, scenario=langChoose)
                            if sst_result is None:
                                data = {"對話梳理": resp_text, "語音識別": "無法識別"}
                                clean_text = "無法識別"
                            else:
                                clean_text = zhconv.convert(_clean_tags(sst_result), 'zh-tw')
                                # clean_resp = re.sub(r'<think>.*?</think>', '', zhconv.convert(resp_text, 'zh-tw'),
                                #                      flags=re.DOTALL).strip()
                                # clean_resp = clean_resp.strip("`").replace("json", "", 1).strip()
                                try:
                                    if isinstance(resp_text, dict):
                                        data = resp_text
                                    else:
                                        # 兜底：尝试从文本提取 JSON
                                        match = re.search(r'\{.*\}', resp_text, re.DOTALL)
                                        if match:
                                            data = json.loads(match.group())
                                        else:
                                            data = {"對話梳理": resp_text}

                                    data['語音識別'] = clean_text
                                    data['Scenario'] = langChoose
                                except Exception as e:
                                    logger.error(f"JSON 解析二次失败: {e}")
                                    data = {"對話梳理": str(resp_text), "語音識別": clean_text}

                    except Exception as e:
                        print(f"處理過程發生異常: {e}")
                        data = {"對話梳理": "音檔問題，無法識別", "語音識別": "無法識別"}
                        clean_text = "無法識別"
                    finally:
                        torch.cuda.empty_cache()
                        if langChoose not in ["IVRTxtBot", "simulate", "text_chat"]:
                            if os.path.exists(WAV_PATH):
                                os.remove(WAV_PATH)

                    # resp_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if langChoose == 'IVRBot':
                        processed_message = json.dumps({
                            "requestId": requestId,
                            "audioFile": audioFile,
                            "lang": langChoose,
                            "createTime": createTime,
                            "asrText": json.dumps(data, ensure_ascii=False),
                            "responseTime": resp_time
                        })
                        target_routing_key = message.reply_to if message.reply_to else MQ_REPLY
                    elif langChoose == 'simulate':
                        processed_message = json.dumps({
                            "requestId": requestId, "sessionId": session_id,
                            "lang": langChoose, "createTime": createTime,
                            "asrText": zhconv.convert(clean_text, 'zh-tw'),
                            "llmResponse": resp_data,
                            "responseTime": resp_time
                        }, ensure_ascii=False)
                        target_routing_key = message.reply_to if message.reply_to else "llm.response"
                    elif langChoose == 'text_chat':
                        processed_message = json.dumps({
                            "requestId": requestId, "sessionId": requestId,
                            "lang": langChoose, "createTime": createTime,
                            "asrText": json.dumps(resp_data, ensure_ascii=False, indent=4),
                            "llmResponse": resp_data,
                            "count": self.bot.get_count(requestId),
                            "responseTime": resp_time
                        }, ensure_ascii=False)
                        target_routing_key = message.reply_to if message.reply_to else "llm.response"
                    else:
                        processed_message = json.dumps({
                            "requestId": requestId, "sessionId": requestId,
                            "lang": langChoose, "createTime": createTime,
                            "asrText": zhconv.convert(clean_text, 'zh-tw'),
                            "llmResponse": resp_data,
                            "count": self.bot.get_count(requestId),
                            "responseTime": resp_time
                        }, ensure_ascii=False)

                        # 3️⃣ 回传 MQ（保持 correlation_id 链路）
                        target_routing_key = message.reply_to if message.reply_to else MQ_REPLY

                    await self.channel.default_exchange.publish(
                        aio_pika.Message(
                            body=processed_message.encode("utf-8"),
                            correlation_id=message.correlation_id,
                            content_type="application/json"
                        ),
                        routing_key=target_routing_key
                    )
                    print(f"[*] message.reply_to = {message.reply_to!r}, MQ_REPLY = {target_routing_key!r}")
                    print(f"[*] publishing to routing_key = {target_routing_key}")
                    print(f"[*] 消息已回传至队列: {target_routing_key} | ID: {message.correlation_id}")

                except Exception as e:
                    logger.error(f"❌ {requestId} 處理異常: {e}")
                    # 显式 NACK 不重回队列，防止毒消息阻塞
                    raise aio_pika.exceptions.MessageProcessError() from e

    async def start(self):
        await self.connect()
        await self.queue.consume(self.process_message)
        logger.info(f"Async Consumer 已啟動 | 並發限流: {MAX_CONCURRENT}")
        await asyncio.Future()  # 永久运行


async def main():
    consumer = RabbitMQConsumer(
        mq_url=MQ_SERVER,
        queue=MQ_QUEUE,
        exchange=MQ_EXCHANGE,
        routing_key=MQ_ROUTE
    )
    await consumer.start()
    


if __name__ == "__main__":

    asyncio.run(main())
