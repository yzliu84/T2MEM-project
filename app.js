'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const steps = [
 {label:'01 / OBSERVE',title:'Observe the scene.',copy:'Extract a compact summary of the current observation.',eq:'Uₜ = Attention(Eₜ, Zₜ)',note:'The memory interface cannot directly attend to action or proprioception tokens.',nodes:['observe','vlm','interface'],paths:['observe','extract'],mode:'observe'},
 {label:'02 / READ',title:'Retrieve before writing.',copy:'Retrieve relevant history from fast weights.',eq:'Qₜ = Uₜ θq  ·  Rₜ = fWₜ(Qₜ)',note:'The current observation has not yet been committed to memory.',nodes:['interface','memory'],paths:['query','read'],mode:'read'},
 {label:'03 / ACT',title:'Combine now with before.',copy:'Combine remembered history with the current scene to choose an action.',eq:'Ũₜ = Uₜ + tanh(α) ⊙ Rₜ',note:'The direct vision-language pathway remains available to the action expert.',nodes:['interface','action'],paths:['fuse','direct','act'],mode:'act'},
 {label:'04 / WRITE',title:'Update the memory.',copy:'A self-supervised update stores the observation for future actions.',eq:'Wₜ₊₁ = Wₜ − wₜ ηₜ βₜ ∇W Lmem',note:'Read-before-write: this update affects subsequent predictions. β ∈ [0.1, 1]; w is the binary write mask.',nodes:['interface','memory','update'],paths:['write','gradient'],mode:'write'}
];
const grid=$('#weight-grid');
function fillTiles(selector,count){const root=$(selector);for(let i=0;i<count;i++){const tile=document.createElement('i');root.appendChild(tile)}}
fillTiles('#weight-grid',36);fillTiles('#backbone-tiles',48);fillTiles('#interface-tiles',12);fillTiles('#action-tiles',12);
let currentStep=0,playing=false,cycle=0,updates=0,phaseStart=0,frame,flowProgress=0;
const phaseDuration=4200;
let flowPaths={};
function paintTiles(selector,seed,mix=0){$$(selector+' i').forEach((tile,i)=>{tile.style.opacity=.25+((i*7+seed*3)%17)/23;tile.style.background=mix&&i%3!==0?'#c1a3ff':'#91baff';})}
function renderFeatures(p){
 const readDone=currentStep>1||(currentStep===1&&p>.65);
 paintTiles('#backbone-tiles',cycle+2);
 paintTiles('#interface-tiles',cycle+(readDone?8:2),readDone);
 $('#interface-tiles').style.opacity=currentStep===0?Math.min(1,.18+p*1.1):1;
 paintTiles('#action-tiles',cycle+8,1);
 $('#action-tiles').style.opacity=currentStep===2?Math.min(1,.15+p*1.5):currentStep===3?1:.2;
 $('#interface-status').textContent=currentStep===0?'Extract observation features':readDone?'Observation + retrieved history':'Query the fast weights';
 grid.querySelectorAll('i').forEach((c,i)=>{const changed=currentStep===3&&p>.55;c.style.opacity=.2+((i*7+(updates+(changed?1:0))*3)%19)/25;c.style.background=changed&&i%3!==0?'#f5b17a':'#a69be5';});
 const routeWindows=currentStep===0?{observe:[0,.25],extract:[.2,.9]}:currentStep===1?{query:[0,.45],read:[.45,.95]}:currentStep===2?{fuse:[0,.75],direct:[0,.75],act:[.75,1]}:{write:[0,.48],gradient:[.48,1]};
 Object.entries(flowPaths).forEach(([name,route])=>{const win=routeWindows[name];route.packets.forEach((tile,j)=>{const q=win?(p-win[0])/(win[1]-win[0]): -1;const t=q*1.4-j*.07;if(t<0||t>1){tile.style.opacity=0;return}const point=route.path.getPointAtLength(t*route.length);tile.setAttribute('x',point.x-4);tile.setAttribute('y',point.y-4);tile.style.opacity=.95;});});
}
function setStep(n){currentStep=n;flowProgress=0;const s=steps[n];$('#step-label').textContent=s.label;$('#step-title').textContent=s.title;$('#step-copy').textContent=s.copy;$('#step-equation').textContent=s.eq;$('#step-note').textContent=s.note;$$('[data-step]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.step===n)));$$('.node').forEach(el=>el.classList.toggle('active',s.nodes.some(k=>el.id==='node-'+k)));$('.flow').dataset.mode=s.mode;$$('[data-path]').forEach(el=>el.classList.toggle('on',s.paths.includes(el.dataset.path)));$('#flow-phase').textContent=['Gather observation → interface','Send query → retrieve history','Fused features → action expert','Observation → self-supervised write'][n];$('#gradient-state').textContent=n===3?'βη∇L → Wₜ₊₁':'L = mean[(fW(K) − V)²]';renderFeatures(playing?0:.8);}
function tick(t){if(!playing)return;if(!phaseStart)phaseStart=t-flowProgress*phaseDuration;flowProgress=Math.min((t-phaseStart)/phaseDuration,1);$('#flow-progress').style.width=flowProgress*100+'%';renderFeatures(flowProgress);if(flowProgress>=1){if(currentStep===3){cycle++;updates++;}setStep((currentStep+1)%4);phaseStart=t;}frame=requestAnimationFrame(tick)}
function setPlaying(value){playing=value;$('#play-flow').textContent=value?'Ⅱ Pause':'▶ Play flow';$('#play-flow').setAttribute('aria-pressed',String(value));$('.flow').classList.toggle('playing',value);cancelAnimationFrame(frame);phaseStart=0;if(value)frame=requestAnimationFrame(tick)}
$$('[data-step]').forEach(b=>b.addEventListener('click',()=>{setPlaying(false);setStep(+b.dataset.step)}));$('#next-step').addEventListener('click',()=>{setPlaying(false);if(currentStep===3){cycle++;updates++;}setStep((currentStep+1)%4)});$('#play-flow').addEventListener('click',()=>setPlaying(!playing));$('#reset-flow').addEventListener('click',()=>{setPlaying(false);cycle=0;updates=0;setStep(0);$('#flow-progress').style.width='0%'});
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
function drawPaths(){
 const svg=$('#flow-lines'),r=$('.flow').getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);
 const box=id=>{const b=$(id).getBoundingClientRect();return {x:b.x-r.x,y:b.y-r.y,w:b.width,h:b.height}};
 const a=box('#node-observe'),v=box('#node-vlm'),i=box('#node-interface'),m=box('#node-memory'),o=box('#node-action'),u=box('#node-update');const cx=b=>b.x+b.w/2,bot=b=>b.y+b.h;
 const paths={observe:`M ${cx(a)} ${bot(a)} L ${cx(v)} ${v.y}`,extract:`M ${cx(v)} ${bot(v)} V ${i.y-38} H ${cx(i)} V ${i.y}`,query:`M ${i.x+i.w} ${i.y+30} L ${m.x} ${m.y+30}`,read:`M ${m.x} ${m.y+70} L ${i.x+i.w} ${i.y+70}`,write:`M ${i.x+i.w} ${bot(i)-15} H ${u.x-18} V ${u.y+u.h/2} H ${u.x}`,fuse:`M ${cx(i)} ${bot(i)} V ${o.y-30} H ${cx(o)} V ${o.y}`,direct:`M ${v.x+v.w} ${v.y+v.h/2} H ${r.width-22} V ${o.y+o.h/2} H ${o.x+o.w}`,act:`M ${cx(o)} ${bot(o)} V ${bot(o)+22}`,gradient:`M ${cx(u)} ${u.y} L ${cx(m)} ${bot(m)}`};
 svg.innerHTML=Object.entries(paths).map(([name,d])=>`<g data-path="${name}"><path class="path-base" d="${d}"/>${Array.from({length:6},()=>`<rect class="flow-packet" width="8" height="8" rx="1" fill="${['write','gradient'].includes(name)?'#f5b17a':name==='read'||name==='fuse'?'#c1a3ff':'#91baff'}"/>`).join('')}</g>`).join('');
 svg.insertAdjacentHTML('beforeend',`<text x="${(i.x+i.w+m.x)/2}" y="${i.y+21}" text-anchor="middle" class="route-label">query →</text><text x="${(i.x+i.w+m.x)/2}" y="${i.y+61}" text-anchor="middle" class="route-label">← history</text>`);
 flowPaths={};$$('#flow-lines [data-path]').forEach(g=>{const path=g.querySelector('path');flowPaths[g.dataset.path]={path,length:path.getTotalLength(),packets:[...g.querySelectorAll('rect')]};g.classList.toggle('on',steps[currentStep].paths.includes(g.dataset.path))});renderFeatures(playing?flowProgress:.8);
}
new ResizeObserver(drawPaths).observe($('.flow'));
setStep(0);if(!matchMedia('(prefers-reduced-motion: reduce)').matches)setPlaying(true);
document.addEventListener('visibilitychange',()=>{if(document.hidden)setPlaying(false)});

if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelectorAll(".storyboard video").forEach(video => { video.autoplay = false; video.pause(); });
}

const tttNS='http://www.w3.org/2000/svg';
function tttElement(tag,attrs,parent){const el=document.createElementNS(tttNS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));parent.appendChild(el);return el}
for(let i=0;i<8;i++){tttElement('rect',{x:48+(i%4)*22,y:177+Math.floor(i/4)*23,width:15,height:15,rx:2,fill:'#91baff',opacity:.45+(i%3)*.23},$('#ttt-interface-grid'));tttElement('rect',{x:807+(i%4)*18,y:179+Math.floor(i/4)*22,width:12,height:12,rx:2,fill:i%2?'#c1a3ff':'#91baff',opacity:0},$('#ttt-context-grid'))}
const tttNodes=[],tttEdges=[];
for(let l=0;l<3;l++){for(let j=0;j<4;j++){const x=460+l*110,y=110+j*43;tttNodes.push({l,j,el:tttElement('circle',{cx:x,cy:y,r:10,fill:'#29314a',stroke:'#737ea4','stroke-width':1.5},$('#ttt-neurons'))});if(l<2)for(let k=0;k<4;k++)tttEdges.push({l,j,k,el:tttElement('path',{d:`M${x} ${y} L${x+110} ${110+k*43}`,stroke:'#596384','stroke-width':1,opacity:.35},$('#ttt-network-edges'))})}}
const tttRouteDefs={q:'M160 186 H420 L460 174',k:'M160 186 H185 V130 H420 L460 174',v:'M160 210 H185 V315 H840 V243',out:'M680 174 H785',pred:'M680 174 H778',back:'M778 205 H700 L680 196 L570 196 L460 196'};
const tttRoutes={};Object.entries(tttRouteDefs).forEach(([name,d])=>{const path=tttElement('path',{d,fill:'none',stroke:'#596384','stroke-width':1.5},$('#ttt-routes'));tttRoutes[name]={path,length:path.getTotalLength(),tiles:Array.from({length:8},()=>tttElement('rect',{width:11,height:11,rx:2,opacity:0},$('#ttt-particles')))}});
tttEdges.forEach(edge=>edge.tile=tttElement('rect',{width:6,height:6,rx:1,opacity:0},$('#ttt-particles')));
let tttMode='read',tttPlaying=!matchMedia('(prefers-reduced-motion: reduce)').matches,tttTime=0,tttLast=0,tttFrame=0;
const tttClamp=x=>Math.max(0,Math.min(1,x));
const tttEase=x=>{const u=tttClamp(x);return u*u*(3-2*u)};
function tttCurve(a,b,c,d,t){const u=1-t;return {x:u*u*u*a.x+3*u*u*t*b.x+3*u*t*t*c.x+t*t*t*d.x,y:u*u*u*a.y+3*u*u*t*b.y+3*u*t*t*c.y+t*t*t*d.y}}
function tttPackets(name,start,end,color){
 const route=tttRoutes[name],q=(tttTime-start)/(end-start);
 route.tiles.forEach((el,i)=>{
  const t=(q-i*.052)/.636;
  if(t<0||t>1){el.setAttribute('opacity',0);return}
  const u=tttEase(t);let pt,size=9,alpha=tttEase(t/.12),angle=0;
  if(name==='out'||name==='pred'){
   const dest=name==='out'?{x:813+(i%4)*18,y:185+Math.floor(i/4)*22}:{x:815+(i%4)*13,y:183+Math.floor(i/4)*19};
   pt=tttCurve({x:680,y:110+(i%4)*43},{x:742,y:110+(i%4)*43},{x:dest.x-35,y:dest.y},dest,u);
   size=5+7*tttEase(t/.5);alpha*=name==='out'?1-tttEase((t-.86)/.14):1-tttEase((t-.75)/.25);
  }else if(name==='back'){
   pt=tttCurve({x:790,y:180+(i%4)*9},{x:758,y:180+(i%4)*9},{x:740,y:165+i*4},{x:720,y:165+i*4},u);
   alpha*=1-tttEase((t-.5)/.5);size=9*(1-.65*tttEase(t));
  }else{
   pt=route.path.getPointAtLength(u*route.length);
   if(name==='q'||name==='k'){
    const blend=tttEase((u-.78)/.22);pt.y+=(110+(i%4)*43-174)*blend;
    alpha*=1-tttEase((t-.82)/.18);size=9*(1-.7*tttEase((t-.72)/.28));
   }else{alpha*=1-tttEase((t-.85)/.15)}
   angle=180*tttEase((pt.x-215)/95);
  }
  const tint=tttEase((pt.x-225)/80),projected=['q','k','v'].includes(name);
  const channel=(hex,j)=>parseInt(hex.slice(1+j*2,3+j*2),16);
  const fill=projected?'rgb('+[0,1,2].map(j=>Math.round(channel('#91baff',j)*(1-tint)+channel(color,j)*tint)).join(',')+')':color;
  el.setAttribute('opacity',alpha);el.setAttribute('fill',fill);
  el.setAttribute('width',size);el.setAttribute('height',size);el.setAttribute('rx',Math.min(size/3,2.5));el.setAttribute('x',pt.x-size/2);el.setAttribute('y',pt.y-size/2);el.setAttribute('transform',`rotate(${angle} ${pt.x} ${pt.y})`);
 });
}
function renderTTT(){
 const write=tttMode==='write',t=tttTime;$('.ttt-lab').dataset.mode=tttMode;
 Object.entries(tttRoutes).forEach(([name,r])=>{r.path.style.display=(write?['k','v','pred','back']:['q','out']).includes(name)?'':'none';r.tiles.forEach(el=>el.setAttribute('opacity',0))});
 tttPackets(write?'k':'q',0,.3,write?'#79d6ca':'#baa5ff');if(write)tttPackets('v',0,.58,'#f1cc79');
 tttPackets(write?'pred':'out',.57,.76,'#c1a3ff');if(write)tttPackets('back',.74,.85,'#f5a975');
 const forward=t>=.26&&t<.63,backward=write&&t>=.77,front=(t-.26)/.37*3,back=(t-.77)/.23*3;
 tttNodes.forEach(({l,j,el})=>{const active=forward&&Math.abs(front-l)<.8||backward&&Math.abs(2-back-l)<.85;el.setAttribute('fill',active?(backward?'#f5a975':'#bca6ff'):backward?'#58413a':'#29314a');el.setAttribute('r',active?12:10)});
 tttEdges.forEach(({l,j,k,el,tile})=>{const travel=backward?back-(1-l):front-l;const moving=forward&&travel>=0&&travel<=1;tile.setAttribute('opacity',moving?.8*tttEase(travel/.15)*(1-tttEase((travel-.8)/.2)):0);if(moving){const u=backward?1-travel:travel;tile.setAttribute('x',460+l*110+110*u-3);tile.setAttribute('y',110+j*43+(k-j)*43*u-3);tile.setAttribute('fill',backward?'#f5a975':'#c1a3ff');}const active=forward&&front>=l&&front<l+1.2||backward&&2-back<=l+1&&2-back>l-.4;el.setAttribute('stroke',backward?'#f5a975':active?'#bca6ff':'#596384');el.setAttribute('opacity',active?.95:.28);el.setAttribute('stroke-width',backward?1+((j+k)%3)*.6:active?2:1)});
 $$('#ttt-context-grid rect').forEach((el,i)=>{const arrival=.57+.19*(i*.052+.636);const blend=tttEase((t-arrival+.022)/.044);el.setAttribute('opacity',!write?.08+blend*(.37+(i%3)*.23):.08)});
 $('#ttt-stage').textContent=t<.27?(write?'1 · Project Uₜ → Kₜ and Vₜ':'1 · Project Uₜ → Qₜ'):t<.64?'2 · Forward through fast weights':!write?'3 · Retrieve memory context Rₜ':t<.77?'3 · Compare prediction with Vₜ':'4 · Backpropagate → update W';
 $('#ttt-network-state').textContent=backward?'Backward pass · Wₜ → Wₜ₊₁':'Forward pass · '+(write?'predict from Kₜ':'Wₜ unchanged');
 $('#ttt-progress').style.width=t*100+'%';
}
function tttTick(now){if(!tttPlaying)return;if(tttLast)tttTime+=(now-tttLast)/8500;tttLast=now;if(tttTime>1.12)tttTime=0;renderTTT();tttFrame=requestAnimationFrame(tttTick)}
function playTTT(value){tttPlaying=value;tttLast=0;cancelAnimationFrame(tttFrame);$('#ttt-play').textContent=value?'Ⅱ Pause':'▶ Play';$('#ttt-play').setAttribute('aria-pressed',String(value));if(value)tttFrame=requestAnimationFrame(tttTick)}
function showTTT(mode){tttMode=mode;tttTime=0;tttLast=0;$$('[data-ttt]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.ttt===mode)));$('#ttt-rule').textContent=mode==='read'?'Rₜ = fWₜ(Uₜ θQ)':'L = mean[(fWₜ(Kₜ) − Vₜ)²]';$('#ttt-rule-note').textContent=mode==='read'?'Read: retrieve history without changing W.':'Wₜ₊₁ = Wₜ − ηeff ∇W L';renderTTT();playTTT(!matchMedia('(prefers-reduced-motion: reduce)').matches)}
$$('[data-ttt]').forEach(b=>b.addEventListener('click',()=>showTTT(b.dataset.ttt)));$('#ttt-play').addEventListener('click',()=>playTTT(!tttPlaying));$('#ttt-replay').addEventListener('click',()=>showTTT(tttMode));
new IntersectionObserver(entries=>{for(const e of entries){if(!e.isIntersecting)playTTT(false);else if(!matchMedia('(prefers-reduced-motion: reduce)').matches)playTTT(true)}},{threshold:.25}).observe($('.ttt-lab'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)playTTT(false)});renderTTT();playTTT(tttPlaying);
