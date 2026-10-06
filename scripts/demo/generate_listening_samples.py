import hashlib,io,json,platform,wave
from pathlib import Path
from importlib.metadata import version
from hifimobinet.inference import Voice
from hifimobinet.registry import manifest
sentences=[
 {'id':'welcome','title':'A short introduction','text':'Welcome to HiFi MobiNet. Choose a voice below and listen to this short demonstration.'},
 {'id':'studio','title':'A longer sentence','text':'At the community studio, we test small speech models and share clear examples with curious listeners.'},
 {'id':'question','title':'A question and a pause','text':'Can a compact speech system make a long sentence sound natural? Listen carefully, then compare the four recordings.'},
]
root=Path('/generated');samples=[]
for model in manifest()['models']:
 voice=Voice(model['id'])
 for sentence in sentences:
  data=voice.synthesize(sentence['text'])
  path=root/sentence['id']/(model['id']+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
  with wave.open(io.BytesIO(data),'rb') as wav:
   assert wav.getframerate()==22050 and wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getnframes()>0
   row={'sentence_id':sentence['id'],'model_id':model['id'],'relative_path':'audio/'+sentence['id']+'/'+model['id']+'.wav','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'sample_rate':wav.getframerate(),'channels':wav.getnchannels(),'frames':wav.getnframes(),'sample_width_bytes':wav.getsampwidth()}
  samples.append(row)
 del voice
record={'schema_version':1,'classification':'new_fixed_demo_audio_not_historical_research_samples','sentences':sentences,'samples':samples,'generation':'One unranked generated output per model and sentence; no selection by quality score.','processing':'Existing Voice.synthesize path with scales [0.667,1.0,0.8] and existing peak-scaled PCM16 conversion. No later waveform edits.','software':{'python':platform.python_version(),'numpy':version('numpy'),'onnxruntime':version('onnxruntime'),'banhmi_phonemize':version('banhmi-phonemize')},'model_repo':'DazielNguyen/hifi-mobiNet-inference','model_revision':'f760e85a45b84c217091acf963837ffc817bad8d','model_identities':{m['id']:{k:m['artifact'][k] for k in ['filename','bytes','sha256']} for m in manifest()['models']},'generated_audio_license':'CC-BY-4.0_only_rights_held_by_team','sentence_basis':'New demo sentences composed for this task; not selected from Harvard.','limits':['No benchmark or quality scoring; no human listening assessment.','Existing Linux amd64 image under local Docker emulation; not target-host inference.','Generated audio does not establish quality equivalence or production readiness.']}
(root/'generation-record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'samples':len(samples),'bytes':sum(x['bytes'] for x in samples),'all_wav_headers_valid':True}))
