'use strict';
const modelDetails = [
  {id:'baseline-resblock2',label:'HiFi-GAN ResBlock2',role:'Internal baseline',badge:'Reference'},
  {id:'parallel-ir',label:'Parallel-IR',role:'Parallel inverted-residual system',badge:'Project model'},
  {id:'sequential-ir',label:'Sequential-IR',role:'Sequential inverted-residual system',badge:'Project model'},
  {id:'piper-original',label:'Piper',role:'External LJSpeech reference',badge:'Reference'}
];
const status = document.getElementById('page-status');
const players = document.getElementById('players');
function element(tag, className, text) {const node=document.createElement(tag);if(className)node.className=className;if(text)node.textContent=text;return node;}
function renderSentence(catalog, sentence) {
  players.querySelectorAll('audio').forEach(player=>{player.pause();player.removeAttribute('src');player.load();});
  players.replaceChildren();
  document.getElementById('sentence-text').textContent=sentence.text;
  document.querySelectorAll('#sentence-tabs button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.sentence===sentence.id)));
  status.textContent='Play one recording at a time. All four models read the same text.';
  for(const model of modelDetails){
    const sample=catalog.samples.find(item=>item.sentence_id===sentence.id && item.model_id===model.id);
    const card=element('article','player-card');card.dataset.model=model.id;
    const top=element('div','card-top');const title=element('div');title.append(element('h3','model-title',model.label),element('span','model-role',model.role));
    top.append(title,element('span','model-badge',model.badge));card.append(top);
    if(!sample){card.append(element('p','player-error','This recording is unavailable.'));players.append(card);continue;}
    const url=new URL(sample.url,window.location.href);
    if(!['https:','http:'].includes(url.protocol))throw new Error('Invalid audio URL');
    const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=url.href;audio.setAttribute('aria-label',`${model.label}: ${sentence.title}`);
    audio.addEventListener('play',()=>{players.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause();});status.textContent=`Now playing ${model.label}.`;});
    audio.addEventListener('ended',()=>{status.textContent='Ready for another recording.';});
    audio.addEventListener('error',()=>{if(!card.querySelector('.player-error'))card.append(element('p','player-error','Audio could not load. Try opening the WAV link below.'));});
    const bottom=element('div','card-bottom');const link=element('a',null,'Open WAV ↗');link.href=url.href;link.target='_blank';link.rel='noopener';
    bottom.append(element('span',null,`${(sample.frames/sample.sample_rate).toFixed(1)} seconds · Mono`),link);card.append(audio,bottom);players.append(card);
  }
}
async function start(){
  try{
    const response=await fetch('catalog.json',{cache:'no-cache'});if(!response.ok)throw new Error('Catalog unavailable');const catalog=await response.json();
    if(catalog.sentences.length!==3 || catalog.samples.length!==12)throw new Error('Incomplete catalog');
    const tabs=document.getElementById('sentence-tabs');
    for(const sentence of catalog.sentences){const button=element('button',null,sentence.title);button.type='button';button.dataset.sentence=sentence.id;button.setAttribute('aria-pressed','false');button.addEventListener('click',()=>renderSentence(catalog,sentence));tabs.append(button);}
    document.getElementById('sample-count').textContent='3 sentences · 12 recordings';renderSentence(catalog,catalog.sentences[0]);
  }catch(error){document.getElementById('sentence-text').textContent='Examples are temporarily unavailable.';status.textContent='Reload the page to try again.';}
}
start();
