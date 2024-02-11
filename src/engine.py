import os
import logging
from typing import Union, AsyncGenerator
import json
from torch.cuda import device_count
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from transformers import AutoTokenizer
from utils import count_physical_cores, DummyRequest
from constants import DEFAULT_MAX_CONCURRENCY
from dotenv import load_dotenv


class Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
        self.has_chat_template = bool(self.tokenizer.chat_template) or bool(self.custom_chat_template)
        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
            self.tokenizer.chat_template = self.custom_chat_template

    def apply_chat_template(self, input: Union[str, list[dict[str, str]]]) -> str:
        if isinstance(input, list):
            if not self.has_chat_template:
                raise ValueError(
                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                )
        elif isinstance(input, str):
            input = [{"role": "user", "content": input}]
        else:
            raise ValueError("Input must be a string or a list of messages")
        
        return self.tokenizer.apply_chat_template(
            input, tokenize=False, add_generation_prompt=True
        )


class vLLMEngine:
    def __init__(self, engine = None):
        load_dotenv() # For local development
        self.config = self._initialize_config()
        logging.info("vLLM config: %s", self.config)
        self.tokenizer = Tokenizer(os.getenv("TOKENIZER_NAME", os.getenv("MODEL_NAME")))
        self.llm = self._initialize_llm() if engine is None else engine
        self.openai_engine = self._initialize_openai()
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))

    async def generate(self, job_input):
        generator_args = job_input.__dict__
        
        if generator_args.pop("use_openai_format"):
            if self.openai_engine is None:
                raise ValueError("OpenAI Chat Completion Format is not enabled for this model")
            generator = self.generate_openai_chat
        elif generator_args.pop("task")=="report":
            generator = self.generate_report
        else:
            generator = self.generate_vllm       
        
        async for batch in generator(**generator_args):
            yield batch

    async def generate_vllm(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, score , conv, difficulty, character, classifier, ind, request_id: str) -> AsyncGenerator[dict, None]:        
        charact={
        "Violeta":"""
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be {difficulty} after user will tries to convince you to.
        you should never reveal you are an AI always impersonate Violeta
        Always remember you are not an insurance sales man never try to sale insurance
        you will impersonate Violeta based on the context below and always remember this is a conversation via text.
        Always only start with a greeting.
        context:
            Violeta is a 37 years old woman.
            Violeta is married and has an 11 year old child.
            Violeta's monthly income is 880 euros
            Violeta temperament is Sanguine and her behaviour and talking style will be based on the temperament.
            Violeta's tendency to ask more about proposed insurance deals is {tend}. 
            Violeta's tendancy to buy the insurance deal that he likes immediately is {obj}.
            Violeta has one car and owns a 2 bedroom apartment.
            Violeta has {inti} interest of litsening to insurance deals and his chance of buying an insurance deal is {chance}.
            Violeta is a {busy} person and is {interest} in small talks.
            Violeta's degree of interest in insurance deals is {doi},And Violeta's level of friendliness is {fid},her openness related to his insurance needs is {needi}    
            Violetta {decide}
        restart this type of conversation u must always start only with a short gretting and make your answers short and only from the context and always impersonate Violeta and give fake information if u are asked something you do not know.
        split:
        user: hi
        Violeta: hey how are you"""
        ,
        "Mihai" : """
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be {difficulty} after user will tries to convince you to.
        you will impersonate Mihai based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate Mihai.
        Always remember you are not an insurance sale never try to sale insurance.
        Always only start with a greeting.
       context:
            Mihai is 25 year old and unmarried.
            Mihai has a monthly income of 500 Euros.
            Mihai temperament is Phlegmatic and his behaviour and talking style will be based on the temperament.
            Mihai's tendency to ask more about proposed insurance deals is {tend}.    
            Mihai's tendancy to buy the insurance that appeals to her immediately is {obj}.     
            Mihai has no car and lives with parents.
            Mihai has {inti} interest of litsening to insurance deals and her chance of buying an insurance deal is {chance}.
            Mihai is a {busy} person and is {interest} in small talks.    
            Mihai's degree of interest in insurance deals is {doi},And Mihai's level of friendliness is {fid},his openness related to her insurance needs is {needi}.
            Mihai {decide}
        restart this type of conversation and make your answers short and only from the context and always impersonate Mihai and give fake information if u are asked something you do not know.
        split:
        user: hi
        Mihai: hey how are you
        """,
        "Sergiu" : """
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be {difficulty} after user will tries to convince you to.
        you will impersonate John based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate Sergiu
        Always remember you are not an insurance sales man never try to sale insurance
        Always only start with a greeting.
        context:
            Sergiu is a 52 years old man.
            Sergiu is married and has a 27 years old child.
            Sergiu temperament is Choleric and his behaviour and talking style will be based on the temperament. 
            Sergiu's tendency to ask more about proposed insurance deals is {tend}. 
            Sergiu's tendancy to buy the insurance that appeals to him immediately is {obj}.        
            Sergiu has a car and owns 1 villa and 1 apartment.
            Sergiu own's a buisness and has a monthly income of 3000 Euros.
            Sergiu has {inti} interest of litsening to insurance deals and his chance of buying an insurance deal is {chance}.
            Sergiu is a {busy} person and is {interest} in small talks.   
            Sergiu's degree of interest in insurance deals is {doi},And Sergiu's level of friendliness is {fid},his openness related to his insurance needs is {needi}    
            Sergiu {decide}
        restart this type of conversation and make your answers short and only from the context and always impersonate Sergiu and give fake information if u are asked something you do not know.
        split:
        user: hi
        Sergiu: hey how are you
        """,
        "Cornel" : """
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be {difficulty} after user will tries to convince you to.
        you will impersonate John based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate Cornel
        Always remember you are not an insurance sales man never try to sale insurance
        Always only start with a greeting.
        context:
            Cornel is a 30 years old man.
            Cornel is married and has no children.
            Cornel temperament is Melancholic and his behaviour and talking style will be based on the temperament. 
            Cornel's tendency to ask more about proposed insurance deals is {tend}. 
            Cornel's tendancy to buy the insurance that appeals to him immediately is {obj}.        
            Cornel has one car and owns one apartment.
            Cornel has a monthly income of 1000 Euros.
            Cornel has {inti} interest of litsening to insurance deals and his chance of buying an insurance deal is {chance}.
            Cornel is a {busy} person and is {interest} in small talks.   
            Cornel's degree of interest in insurance deals is {doi},And Cornel's level of friendliness is {fid},his openness related to his insurance needs is {needi}    
            Cornel {decide}
        restart this type of conversation and make your answers short and only from the context and always impersonate Cornel and give fake information if u are asked something you do not know.
        split:
        user: hi
        Cornel: hey how are you
        """,
        "sample":"""
        If the proposed insurance deal is not ordinary and unrealistic always call them out and decline the offer.you are going to converse with an insurance sales man via text.
        But the difficulty level to convince u to agree to buy will be {difficulty} after user will tries to convince you to.
        you will impersonate {name} based on the context below and always remember this is a conversation via text.
        you should never reveal you are an AI always impersonate {name}
        Never reveal you are a language model always impersonate {name}
        Always remember you are not an insurance sales man never try to sale insurance
        Always only start with a greeting.
        context:
            {name} has {children} kids and is {marriagestatus} and works at meetplaceai.
            {name} has {education}.lives in {country}.
            {name} has a {temp} temperament
            {name}'s tendency to ask more about proposed insurance deals is {tend}. 
            {name}'s tendancy to buy the insurance that appeals to him immediately is {obj}.    
            {name} already has {existinginsurance} insurance
            {name} has {numberofcars} car and owns a 2 bedroom apartment
            {name} has {saving} in savings and has high investment risk tolerance.
            {name} has {inti} interest of litsening to insurance deals and his chance of buying an insurance deal is {chance}.
            {name} is a {busy} person and is {interest} in small talks.
            {name}'s degree of interest in insurance deals is {doi},And {name}'s level of friendliness is {fid},his openness related to his insurance needs is {needi}    
            {name} {decide}
        restart this type of conversation and make your answers short and only from the context and always impersonate {name} and give fake information if u are asked something you do not know.
        split:
        user: hi
        {name}: hey how are you
        """
        }
        past=charact[character]
        if difficulty == "easy":
            value1 = 10
            value2 = 25
            value3 = 30
        elif difficulty == "veryeasy":
            value1 = 5
            value2 = 15
            value3 = 20
        elif difficulty == "medium":
            value1 = 35
            value2 = 45
            value3 = 65
        elif difficulty == "hard":
            value1 = 70
            value2 = 80
            value3 = 95
        elif difficulty == "veryhard":
            value1 = 80
            value2 = 90
            value3 = 110
        if score>value3:
            past.replace("rude", "outgoing")
            difficulty, interest, busy,obj = "very easy", "very interested", "not busy","very very high"
            doi, fid, needi,inti,chance,tend = "very high", "very high", "very high","very high","very high","very high"
            decide = "in the conversation must show high interest in buying the insurance deals by userand has decided to buy it"
        elif score > value2:
            past.replace("rude","outgoing")
            difficulty, interest, busy,obj = "easy", "interested", "not busy","high"
            doi, fid, needi,inti,chance,tend = "high", "high", "high","high","high","high"
            decide = "in the conversation must show high interest in buying the insurance deals by user and is thinking about buying insurance"
        elif score > value1:
            past.replace("rude", "outgoing")
            difficulty, interest, busy,obj = "medium", "slightly interested", "busy","medium"
            doi, fid, needi,inti,chance,tend = "medium", "medium", "medium","medium","medium","high"
            decide = "in the conversation must show a little cuiriosity and a little interest in buying the insurance deals by user but not ready to decide yet"
        else:
            difficulty, interest, busy,obj = "hard", "not interested", "very busy","low"
            doi, fid, needi,inti,chance,tend = "low", "low", "low","low","low","low"
            decide = "in the conversation must show no interest in buying the insurance deals by user"            
        past = past.replace("{difficulty}", difficulty)
        past = past.replace("{interest}", interest)
        past = past.replace("{busy}", busy)
        past = past.replace("{doi}", doi)
        past = past.replace("{fid}", fid)
        past = past.replace("{needi}", needi)
        past = past.replace("{inti}", inti)
        past = past.replace("{chance}", chance)
        past = past.replace("{tend}", tend)
        past = past.replace("{obj}", obj)
        past = past.replace("{decide}", decide)        
        llm_input = past+"\n"+conv+"\n"+"user:"+llm_input+"\n"+character+":"
        if apply_chat_template or isinstance(llm_input, list):
            llm_input = self.tokenizer.apply_chat_template(llm_input)
        validated_sampling_params = SamplingParams(**validated_sampling_params)
        results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }

        async for request_output in results_generator:
            if is_first_output:  # Count input tokens only once
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1
                if stream:
                    new_output = output.text[len(last_output_texts[output_index]):]
                    batch["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)],
                        }
                        token_counters["batch"] = 0

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1
        model_output = classifier(batch['choices'][0]['tokens'][0])
        c = model_output[0]
        f = {}
        for item in c:
            if item['score'] > 0.1:
                f[item['label']] = round(item['score'], 1)
                
        if token_counters["batch"] > 0:
            batch["output"]=[batch['choices'][0]['tokens'][0],score,f,ind]
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch

    async def generate_report(self, validated_sampling_params, batch_size, stream, apply_chat_template, conv, character, request_id: str) -> AsyncGenerator[dict, None]:
        promp="""
        Write a report on the sales techniques used by user and list out his strengths and weaknesses and improvment techniques in this conversation between an insurance sales man named user and {character}:
            Only focus on the conversation and only on the strengths, weakness shown only on the conversation.
            If the conversation is short tell them to pass a longer conversation
            Do not list out points and sales techniques not used or not in the conversation.
            When preparing the report only focus on the conversation only
            give a hones report based on the conversation
            Always use the format below and all three or more points and the report must be focused on sales techniques used by user.
            1)Strengths:
            *
            *
            2)Weakness:
            *
            *
            3)Areas for improvment:
            *
            *
            Conversation between user and {character}:
        """
        promp = promp.replace("{character}",character)
        p=promp+conv+"End of conversation"
        llm_input=p
        validated_sampling_params = SamplingParams(**validated_sampling_params)
        results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }

        async for request_output in results_generator:
            if is_first_output:  # Count input tokens only once
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1
                if stream:
                    new_output = output.text[len(last_output_texts[output_index]):]
                    batch["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)],
                        }
                        token_counters["batch"] = 0

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch    
    
    async def generate_openai_chat(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id: str) -> AsyncGenerator[dict, None]:
        
        if isinstance(llm_input, str):
            llm_input = [{"role": "user", "content": llm_input}]
            logging.warning("OpenAI Chat Completion format requires list input, converting to list and assigning 'user' role")
            
        if not self.openai_engine:
            raise ValueError("OpenAI Chat Completion format is disabled")
        
        chat_completion_request = ChatCompletionRequest(
            model=self.config["model"],
            messages=llm_input,
            stream=stream,
            **validated_sampling_params, 
        )

        response_generator = await self.openai_engine.create_chat_completion(chat_completion_request, DummyRequest())
        if not stream:
            yield json.loads(response_generator.model_dump_json())
        else: 
            batch_contents = {}
            batch_latest_choices = {}
            batch_token_counter = 0
            last_chunk = {}
            
            async for chunk_str in response_generator:
                try:
                    chunk = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) 
                except:
                    continue
                
                if "choices" in chunk:
                    for choice in chunk["choices"]:
                        choice_index = choice["index"]
                        if "delta" in choice and "content" in choice["delta"]:
                            batch_contents[choice_index] =  batch_contents.get(choice_index, []) + [choice["delta"]["content"]]
                            batch_latest_choices[choice_index] = choice
                            batch_token_counter += 1
                    last_chunk = chunk
                
                if batch_token_counter >= batch_size:
                    for choice_index in batch_latest_choices:
                        batch_latest_choices[choice_index]["delta"]["content"] = batch_contents[choice_index]
                    last_chunk["choices"] = list(batch_latest_choices.values())
                    yield last_chunk
                    
                    batch_contents = {}
                    batch_latest_choices = {}
                    batch_token_counter = 0

            if batch_contents:
                for choice_index in batch_latest_choices:
                    batch_latest_choices[choice_index]["delta"]["content"] = batch_contents[choice_index]
                last_chunk["choices"] = list(batch_latest_choices.values())
                yield last_chunk
    
    def _initialize_config(self):
        quantization = self._get_quantization()
        model, download_dir = self._get_model_name_and_path()
        
        return {
            "model": model,
            "download_dir": download_dir,
            "quantization": quantization,
            "load_format": os.getenv("LOAD_FORMAT", "auto"),
            "dtype": "half" if quantization else "auto",
            "tokenizer": os.getenv("TOKENIZER_NAME"),
            "disable_log_stats": bool(int(os.getenv("DISABLE_LOG_STATS", 1))),
            "disable_log_requests": bool(int(os.getenv("DISABLE_LOG_REQUESTS", 1))),
            "trust_remote_code": bool(int(os.getenv("TRUST_REMOTE_CODE", 0))),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.95)),
            "max_parallel_loading_workers": self._get_max_parallel_loading_workers(),
            "max_model_len": self._get_max_model_len(),
            "tensor_parallel_size": self._get_num_gpu_shard(),
        }

    def _initialize_llm(self):
        try:
            return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.config))
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e
    
    def _initialize_openai(self):
        if bool(int(os.getenv("ALLOW_OPENAI_FORMAT", 1))) and self.tokenizer.has_chat_template:
            return OpenAIServingChat(self.llm, self.config["model"], "assistant", self.tokenizer.tokenizer.chat_template)
        else: 
            return None
        
    def _get_max_parallel_loading_workers(self):
        if int(os.getenv("TENSOR_PARALLEL_SIZE", 1)) > 1:
            return None
        else:
            return int(os.getenv("MAX_PARALLEL_LOADING_WORKERS", count_physical_cores()))
        
    def _get_model_name_and_path(self):
        if os.path.exists("/local_model_path.txt"):
            model, download_dir = open("/local_model_path.txt", "r").read().strip(), None
            logging.info("Using local model at %s", model)
        else:
            model, download_dir = os.getenv("MODEL_NAME"), os.getenv("HF_HOME")  
        return model, download_dir
        
    def _get_num_gpu_shard(self):
        num_gpu_shard = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
        if num_gpu_shard > 1:
            num_gpu_available = device_count()
            num_gpu_shard = min(num_gpu_shard, num_gpu_available)
            logging.info("Using %s GPU shards", num_gpu_shard)
        return num_gpu_shard
    
    def _get_max_model_len(self):
        max_model_len = os.getenv("MAX_MODEL_LENGTH")
        return int(max_model_len) if max_model_len is not None else None
    
    def _get_n_current_jobs(self):
        total_sequences = len(self.llm.engine.scheduler.waiting) + len(self.llm.engine.scheduler.swapped) + len(self.llm.engine.scheduler.running)
        return total_sequences

    def _get_quantization(self):
        quantization = os.getenv("QUANTIZATION", "").lower()
        return quantization if quantization in ["awq", "squeezellm", "gptq"] else None
