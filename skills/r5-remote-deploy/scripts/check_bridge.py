#!/usr/bin/env python3
"""One explicitly requested deployment check; makes ONE real multimodal model call."""
import base64
import io
import json
import os
from pathlib import Path
from PIL import Image
from robot_tasks.policies.codex_bridge import create_codex_client
from robot_tasks.policies.model_config import load_gpt_settings

image=io.BytesIO();Image.new('RGB',(32,32),(255,0,0)).save(image,format='PNG')
settings=load_gpt_settings(Path('configs/models/gpt.toml'))
client=create_codex_client(settings);client.require_chatgpt_login()
schema={'type':'object','properties':{'color':{'type':'string','enum':['red','green','blue']}},'required':['color'],'additionalProperties':False}
reply=client.generate(instructions='Identify the solid color of the attached test image. Return the schema JSON only.',messages=[{'role':'user','content':[{'type':'input_text','text':'What is the image color?'},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(image.getvalue()).decode()}]}],schema=schema,on_reasoning=lambda text:print('BRIDGE_SUMMARY',text,flush=True))
result={'host':os.uname().nodename,'passed':json.loads(reply.text)=={'color':'red'},'model':reply.model,'reasoning_effort':settings.reasoning_effort,'service_tier':settings.service_tier,'latency_s':reply.latency_s,'usage':reply.usage,'model_calls':1,'image_verified':True}
root=Path(__file__).resolve().parents[3]
out=Path(os.environ.get('GPT_BENCHMARK_RUNTIME_ROOT',root.parent/'runtime_combined'))/'bridge_checks'
out.mkdir(parents=True,exist_ok=True);(out/(os.uname().nodename+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 2)
