'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const steps = [
 {label:'01 / OBSERVE',title:'Observe the scene.',copy:'Extract a compact summary of the current observation.',eq:'Uₜ = Attention(Eₜ, Zₜ)',note:'The memory interface cannot directly attend to action or proprioception tokens.',nodes:['observe','vlm','interface'],paths:['observe','extract'],mode:'observe'},
 {label:'02 / READ',title:'Retrieve before writing.',copy:'Retrieve relevant history from fast weights.',eq:'Qₜ = Uₜ θq  ·  Rₜ = fWₜ(Qₜ)',note:'The current observation has not yet been committed to memory.',nodes:['interface','memory'],paths:['query','read'],mode:'read'},
 {label:'03 / ACT',title:'Combine now with before.',copy:'Combine remembered history with the current scene to choose an action.',eq:'Ũₜ = Uₜ + tanh(α) ⊙ Rₜ',note:'The direct vision-language pathway remains available to the action expert.',nodes:['interface','action'],paths:['fuse','direct','act'],mode:'act'},
 {label:'04 / WRITE',title:'Update the memory.',copy:'A self-supervised update stores the observation for future actions.',eq:'Wₜ₊₁ = Wₜ − wₜ ηₜ βₜ ∇W Lmem',note:'Read-before-write: this update affects subsequent predictions. β ∈ [0.1, 1]; w is the binary write mask.',nodes:['interface','memory','update'],paths:['write','gradient'],mode:'write'}
];
const grid=$('#weight-grid'); for(let i=0;i<25;i++){const cell=document.createElement('i');cell.style.opacity=.18+((i*7+3)%13)/17;grid.appendChild(cell)}
let currentStep=0,playing=false,cycle=0,updates=0,phaseStart=0,frame;
function setStep(n){if(n===3&&currentStep!==3)updates++;currentStep=n;const s=steps[n];$('#step-label').textContent=s.label;$('#step-title').textContent=s.title;$('#step-copy').textContent=s.copy;$('#step-equation').textContent=s.eq;$('#step-note').textContent=s.note;$$('[data-step]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.step===n)));$$('.node').forEach(el=>el.classList.toggle('active',s.nodes.some(k=>el.id==='node-'+k)));$('.flow').dataset.mode=s.mode;$$('[data-path]').forEach(el=>el.classList.toggle('on',s.paths.includes(el.dataset.path)));$('#flow-phase').textContent=['Observation → interface','Query → memory → readout','Context → action expert','K/V → loss → weight update'][n];$('#gradient-state').textContent=n===3?'βη∇L → Wₜ₊₁':'L = mean[(fW(K) − V)²]';grid.querySelectorAll('i').forEach((c,i)=>{c.style.opacity=.18+((i*7+3+updates*3)%13)/17;c.classList.toggle('writing',n===3)});}
function tick(t){if(!playing)return;if(!phaseStart)phaseStart=t;const elapsed=t-phaseStart;$('#flow-progress').style.width=Math.min(elapsed/2600,1)*100+'%';if(elapsed>=2600){phaseStart=t;if(currentStep===3)cycle++;setStep((currentStep+1)%4)}frame=requestAnimationFrame(tick)}
function setPlaying(value){playing=value;$('#play-flow').textContent=playing?'Ⅱ Pause':'▶ Play flow';$('#play-flow').setAttribute('aria-pressed',String(playing));$('.flow').classList.toggle('playing',playing);cancelAnimationFrame(frame);phaseStart=0;if(playing)frame=requestAnimationFrame(tick)}
$$('[data-step]').forEach(b=>b.addEventListener('click',()=>{setPlaying(false);setStep(+b.dataset.step)}));$('#next-step').addEventListener('click',()=>{setPlaying(false);if(currentStep===3)cycle++;setStep((currentStep+1)%4)});$('#play-flow').addEventListener('click',()=>setPlaying(!playing));$('#reset-flow').addEventListener('click',()=>{setPlaying(false);cycle=0;updates=0;setStep(0);$('#flow-progress').style.width='0%'});
const phaseData=[['Trainable','Not present','Learn the task.','Train the policy without memory.'],['Frozen','Trainable','Learn what to remember.','Freeze the policy; action supervision trains memory.'],['Trainable','Frozen','Learn how to use memory.','Freeze memory’s slow parameters; train the policy to use history.']];
let trainingPhase=0, trainingPlaying=false, trainingFrame=0, trainingStart=0;
function phase(n){
 trainingPhase=n;const d=phaseData[n];
 $('#policy-state').textContent=d[0];$('#memory-state').textContent=d[1];
 $('#phase-title').textContent=d[2];$('#phase-copy').textContent=d[3];
 $$('[data-phase]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.phase===n)));
 $('.training').dataset.phase=String(n);
 $('#training-phase-label').textContent=['Stage 1 · No memory','Stage 2A · Memory learning','Stage 2B · Policy learning'][n];
 $('#gradient-caption').textContent=n===1?'Action gradients pass through the fixed policy to train memory':'Action supervision trains policy slow parameters';
 $('#training-gradient-path').setAttribute('d',n===1?'M500 30 V10 H100 V30':'M500 30 V10 H300 V30');
 $('#training-memory').classList.toggle('is-learning',n===1);
 $('#training-policy').classList.toggle('is-learning',n!==1);
 $('#training-memory').classList.toggle('is-absent',n===0);
 $('#training-memory > span').textContent=n===0?'No memory':'Memory';
 $('#training-memory > small').textContent=n===0?'Introduced in Stage 2':'Slow parameters';
 $('.training-key').textContent=n===0?'Blue: policy learning':'Blue: parameter learning · Orange: online memory updates';
 $('#training-episode').classList.toggle('is-updating',n!==0);
 $('#episode-label').textContent=n===0?'No memory reads, writes or fast-weight updates':'Fast weights update within each episode';
}
function trainingTick(t){
 if(!trainingPlaying)return;
 if(!trainingStart)trainingStart=t;
 const progress=(t-trainingStart)/4500;
 $('#training-progress').style.width=Math.min(progress,1)*100+'%';
 if(progress>=1){phase(trainingPhase===2?1:trainingPhase+1);trainingStart=t;}
 trainingFrame=requestAnimationFrame(trainingTick);
}
function playTraining(value){
 trainingPlaying=value;cancelAnimationFrame(trainingFrame);trainingStart=0;
 $('.training').classList.toggle('is-playing',value);
 $('#training-play').textContent=value?'Ⅱ Pause':'▶ Play';
 $('#training-play').setAttribute('aria-pressed',String(value));
 if(value)trainingFrame=requestAnimationFrame(trainingTick);
}
$$('[data-phase]').forEach(b=>b.addEventListener('click',()=>{playTraining(false);phase(+b.dataset.phase);$('#training-progress').style.width='0%';}));
$('#training-play').addEventListener('click',()=>playTraining(!trainingPlaying));
$('#training-next').addEventListener('click',()=>{playTraining(false);phase(trainingPhase===2?1:trainingPhase+1);$('#training-progress').style.width='0%';});
$('#training-reset').addEventListener('click',()=>{playTraining(false);phase(0);$('#training-progress').style.width='0%';});
phase(0);
let trainingAutoStarted=false;
const trainingObserver=new IntersectionObserver(entries=>{
 for(const entry of entries){
  if(entry.isIntersecting&&!trainingAutoStarted){trainingAutoStarted=true;if(!matchMedia('(prefers-reduced-motion: reduce)').matches)playTraining(true);}
  else if(!entry.isIntersecting&&trainingPlaying)playTraining(false);
 }
},{threshold:.35});
trainingObserver.observe($('.training'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)playTraining(false);});
const tasks=['BinFill','PickXtimes','SwingXtimes','StopCube','VideoUnmask','ButtonUnmask','VideoUnmaskSwap','ButtonUnmaskSwap','PickHighL','VideoRepick','VideoPlcBtn','VideoPlcOrd','MoveCube','InsertPeg','PatternLock','RouteStick'];
const categories=['Counting','Permanence','Reference','Imitation'];
const methods=[{name:'T²Mem',avg:56.83,scores:[60.67,91.33,90.67,70,88,59.33,29.33,20,50.67,38.67,35.33,30,80.67,36,50,78.67]}, {name:'FrameSamp + Modul',avg:44.51,scores:[39.56,87.33,92,42,32.67,25.11,24.44,18.22,22.89,30.44,60,32,77.78,7.56,53.56,66.67]}, {name:'MemER',avg:42.38,scores:[56.67,79.33,59.33,0,81.33,72,38,21.33,70.67,25.33,30,26,82.67,6.67,16.67,12]}, {name:'π₀.₅ · no memory',avg:17.93,scores:[30,42.89,35.56,6.67,20.44,22.22,18.67,6.67,11.33,.44,31.11,25.78,26,1.56,2.89,4.67]}];
let category='All';
function indices(){return tasks.map((_,i)=>i).filter(i=>category==='All'||categories[Math.floor(i/4)]===category)}
function renderResults(){const idx=indices();const task=$('#task-select').value;$('#chart-title').textContent=task==='average'?(category==='All'?'All tasks':category)+' · mean success':tasks[+task]+' · success';$('#result-bars').innerHTML=methods.map((m,i)=>{const value=task!=='average'?m.scores[+task]:category==='All'?m.avg:idx.reduce((s,j)=>s+m.scores[j],0)/idx.length;return `<div class="result-row ${i===0?'ours':''}"><span class="result-name">${m.name}</span><div class="result-track"><div class="result-fill" style="width:${value}%"></div></div><span class="result-value">${value.toFixed(2)}</span></div>`}).join('');}
function changeCategory(c){category=c;$$('[data-category]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.category===c)));$('#task-select').innerHTML='<option value="average">'+(c==='All'?'All-task average':'Category average')+'</option>'+indices().map(i=>`<option value="${i}">${tasks[i]}</option>`).join('');renderResults()}
$$('[data-category]').forEach(b=>b.addEventListener('click',()=>changeCategory(b.dataset.category)));$('#task-select').addEventListener('change',renderResults);changeCategory('All');
$('#task-table').innerHTML=tasks.map((task,i)=>`<tr><th scope="row">${task}</th>${[3,2,1,0].map(m=>`<td>${methods[m].scores[i].toFixed(2)}</td>`).join('')}</tr>`).join('');
const memories=[{score:100,count:300,title:'The right history guides the action.',copy:'Matching memory restores the hidden cue.'},{score:100/300,count:1,title:'The wrong history misleads the policy.',copy:'Wrong history is worse than no history.'},{score:101/3,count:101,title:'An empty memory leaves the cue missing.',copy:'Near the one-third chance level.'}];
const memoryStories=[
 ['Same episode','The demonstration identifies target A.','Matching history','A','Fast weights formed from that demonstration.','Correct: use memory built from this episode’s own demonstration.'],
 ['Conflicting episode','The donor demonstration identifies target B.','Conflicting history','B','Replace the memory with the donor’s fast weights.','Conflicting: give the policy memory from a different episode with a conflicting target. The target episode still requires A.'],
 ['No demonstration','Skip demonstration-based memory formation.','Initial memory W₀','W₀','Keep the learned initialization; no demonstration history.','Empty: use the learned initial memory W₀ without a demonstration. The weights are not all zero.']
];
const memoryChoices=[
 ['Remember A → choose A','The remembered cue agrees with the true target.','✓ Correct target'],
 ['Remember B → misled toward B','The donor cue points away from the true target A.','× Misled by history'],
 ['No cue → guess','Each draw selects A, B or C. This illustrates guessing, not measured policy probabilities.','? Random choice']
];
let emptyChoiceTimer, lastEmptyChoice=-1;
function drawEmptyChoice(){
 if(document.hidden)return;
 const choice=lastEmptyChoice<0?Math.floor(Math.random()*3):(lastEmptyChoice+1+Math.floor(Math.random()*2))%3;
 lastEmptyChoice=choice;
 $$('.choice-target').forEach((el,i)=>el.classList.toggle('is-guessed',i===choice));
 $('#choice-result').textContent=(choice===0?'✓ A · correct by chance':'× '+['A','B','C'][choice]+' · wrong target');
}
function memory(n){clearInterval(emptyChoiceTimer);lastEmptyChoice=-1;$$('.choice-target').forEach(el=>el.classList.remove('is-guessed'));const m=memories[n],story=memoryStories[n];['choice-title',null,'choice-result'].forEach((id,i)=>{if(id)$('#'+id).textContent=memoryChoices[n][i]});$('#memory-explainer').dataset.condition=String(n);['source-title',null,'stored-title','stored-cue',null,'memory-definition'].forEach((id,i)=>{if(id)$('#'+id).textContent=story[i]});$('#memory-score').textContent=m.score===100?'100':m.score.toFixed(2);$('#memory-count').textContent=m.count+' / 300 successful trials';$('#memory-bar').style.width=m.score+'%';$('#memory-title').textContent=m.title;$('#memory-copy').textContent=m.copy;$$('[data-memory]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.memory===n)));if(n===2){drawEmptyChoice();if(!matchMedia('(prefers-reduced-motion: reduce)').matches)emptyChoiceTimer=setInterval(drawEmptyChoice,600);}}$$('[data-memory]').forEach(b=>b.addEventListener('click',()=>memory(+b.dataset.memory)));memory(0);
function drawPaths(){const svg=$('#flow-lines'),r=$('.flow').getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);const box=id=>{const b=$(id).getBoundingClientRect();return {x:b.x-r.x,y:b.y-r.y,w:b.width,h:b.height}};const a=box('#node-observe'),v=box('#node-vlm'),i=box('#node-interface'),m=box('#node-memory'),o=box('#node-action'),u=box('#node-update');const cx=b=>b.x+b.w/2,bot=b=>b.y+b.h;const paths={observe:`M ${cx(a)} ${bot(a)} L ${cx(v)} ${v.y}`,extract:`M ${cx(v)} ${bot(v)} V ${i.y-10} H ${cx(i)} V ${i.y}`,query:`M ${i.x+i.w} ${i.y+i.h*.28} L ${m.x} ${m.y+m.h*.28}`,read:`M ${m.x} ${m.y+m.h*.55} L ${i.x+i.w} ${i.y+i.h*.55}`,write:`M ${i.x+i.w} ${i.y+i.h*.82} L ${m.x} ${m.y+m.h*.82}`,fuse:`M ${cx(i)} ${bot(i)} V ${o.y-12} H ${cx(o)} V ${o.y}`,direct:`M ${v.x+v.w} ${v.y+v.h/2} H ${r.width-16} V ${o.y+o.h/2} H ${o.x+o.w}`,act:`M ${cx(o)} ${bot(o)} V ${bot(o)+25}`,gradient:`M ${cx(m)} ${bot(m)} V ${u.y} M ${u.x+u.w} ${u.y+u.h/2} H ${r.width-29} V ${m.y+m.h*.7} H ${m.x+m.w}`};svg.innerHTML='<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#91a8fa"/></marker><marker id="write-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#f5b17a"/></marker></defs>'+Object.entries(paths).map(([name,d])=>`<g data-path="${name}"><path class="path-base" d="${d}" marker-end="url(#${['write','gradient'].includes(name)?'write-arrow':'arrow'})"/><path class="path-pulse" d="${d}"/></g>`).join('');setStep(currentStep)}
new ResizeObserver(drawPaths).observe($('.flow'));
setStep(0);if(!matchMedia('(prefers-reduced-motion: reduce)').matches)setPlaying(true);
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false)});

if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelectorAll(".storyboard video").forEach(video => { video.autoplay = false; video.pause(); });
}
