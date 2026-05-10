// ---- MLP State ----
let nIn=2,nHid=2,nOut=1;
let actFn='sigmoid';
let inputs=[];
let W1=[],b1=[],W2=[],b2=[];
let epoch=0,lossHistory=[],training=false,trainInterval=null;
let lr=0.1;

// ---- Activation functions ----
function sigmoid(x){return 1/(1+Math.exp(-x))}
function relu(x){return Math.max(0,x)}
function tanh_(x){return Math.tanh(x)}
function leaky(x){return x>0?x:0.01*x}
function activate(x){
  if(actFn==='sigmoid')return sigmoid(x);
  if(actFn==='relu')return relu(x);
  if(actFn==='tanh')return tanh_(x);
  return leaky(x);
}
function activateDeriv(a){
  if(actFn==='sigmoid')return a*(1-a);
  if(actFn==='relu')return a>0?1:0;
  if(actFn==='tanh')return 1-a*a;
  return a>0?1:0.01;
}

// ---- Network init ----
function randW(){return(Math.random()-0.5)*2}
function initNet(){
  W1=[];b1=[];W2=[];b2=[];
  for(let i=0;i<nHid;i++){
    W1.push([]);
    for(let j=0;j<nIn;j++)W1[i].push(randW());
    b1.push(randW());
  }
  for(let i=0;i<nOut;i++){
    W2.push([]);
    for(let j=0;j<nHid;j++)W2[i].push(randW());
    b2.push(randW());
  }
  epoch=0;lossHistory=[];
  updateStats(0,null,null);
  drawLoss();drawDecision();updateXORTable();
}

// ---- Forward pass ----
function forward(inp){
  let h=[];
  for(let i=0;i<nHid;i++){
    let z=b1[i];
    for(let j=0;j<nIn;j++)z+=W1[i][j]*inp[j];
    h.push(activate(z));
  }
  let o=[];
  for(let i=0;i<nOut;i++){
    let z=b2[i];
    for(let j=0;j<nHid;j++)z+=W2[i][j]*h[j];
    o.push(sigmoid(z));
  }
  return{h,o};
}

function forwardRaw(inp){
  let hz=[];
  for(let i=0;i<nHid;i++){
    let z=b1[i];
    for(let j=0;j<nIn;j++)z+=W1[i][j]*inp[j];
    hz.push(z);
  }
  let h=hz.map(activate);
  let oz=[];
  for(let i=0;i<nOut;i++){
    let z=b2[i];
    for(let j=0;j<nHid;j++)z+=W2[i][j]*h[j];
    oz.push(z);
  }
  let o=oz.map(sigmoid);
  return{hz,h,oz,o};
}

// ---- XOR dataset ----
const xorData=[[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]];

// ---- Training ----
function trainOnce(){
  let totalLoss=0;
  for(let[inp,tgt]of xorData){
    let{hz,h,oz,o}=forwardRaw(inp);
    let dO=[];
    for(let i=0;i<nOut;i++){
      let e=o[i]-tgt[i];
      totalLoss+=e*e;
      dO.push(e*o[i]*(1-o[i]));
    }
    let dH=Array(nHid).fill(0);
    for(let i=0;i<nOut;i++)
      for(let j=0;j<nHid;j++)
        dH[j]+=dO[i]*W2[i][j];
    let dHz=dH.map((d,j)=>d*activateDeriv(h[j]));
    for(let i=0;i<nOut;i++){
      for(let j=0;j<nHid;j++)W2[i][j]-=lr*dO[i]*h[j];
      b2[i]-=lr*dO[i];
    }
    for(let i=0;i<nHid;i++){
      for(let j=0;j<nIn;j++)W1[i][j]-=lr*dHz[i]*inp[j];
      b1[i]-=lr*dHz[i];
    }
  }
  return totalLoss/xorData.length;
}

function computeAcc(){
  let c=0;
  for(let[inp,tgt]of xorData){
    let{o}=forward(inp);
    if(Math.round(o[0])===tgt[0])c++;
  }
  return Math.round(c/xorData.length*100);
}

function trainStep(){
  for(let i=0;i<10;i++){
    let l=trainOnce();epoch++;
    if(epoch%2===0)lossHistory.push(l);
  }
  let loss=lossHistory[lossHistory.length-1]||0;
  let acc=computeAcc();
  updateStats(epoch,loss,acc);
  drawLoss();drawDecision();drawNet();updateFormula();updateXORTable();
}

function toggleTrain(){
  training=!training;
  let btn=document.getElementById('train-btn');
  if(training){
    btn.textContent='⏸ Pause';
    trainInterval=setInterval(()=>{
      for(let i=0;i<8;i++){let l=trainOnce();epoch++;if(epoch%3===0)lossHistory.push(l);}
      let loss=lossHistory[lossHistory.length-1]||0;
      let acc=computeAcc();
      updateStats(epoch,loss,acc);
      drawLoss();drawDecision();drawNet();updateFormula();updateXORTable();
    },80);
  }else{
    btn.textContent='▶ Train';
    clearInterval(trainInterval);
  }
}

function resetNet(){
  if(training)toggleTrain();
  initNet();
  setupInputSliders();
  drawNet();drawActFn();updateFormula();
}

function updateStats(ep,loss,acc){
  document.getElementById('s-epoch').textContent=ep;
  document.getElementById('s-loss').textContent=loss!=null?loss.toFixed(4):'—';
  document.getElementById('s-acc').textContent=acc!=null?acc+'%':'—';
  if(loss!=null){
    let pct=Math.max(0,Math.min(100,(1-loss)*100));
    let bar=document.getElementById('loss-bar');
    bar.style.width=pct+'%';
    bar.style.background=acc===100?'#1D9E75':'#7F77DD';
  }
}

function updateLR(v){
  lr=parseFloat(v)*0.1;
  document.getElementById('lr-val').textContent=lr.toFixed(1);
  document.getElementById('s-lr').textContent=lr.toFixed(1);
}

// ---- Architecture / Activation switches ----
function setArch(ni,nh,no,el){
  nIn=ni;nHid=nh;nOut=no;
  document.querySelectorAll('#arch-tabs .tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  initNet();setupInputSliders();drawNet();drawActFn();updateFormula();drawLoss();drawDecision();
}

function setAct(fn,el){
  actFn=fn;
  document.querySelectorAll('#act-tabs .tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  drawActFn();updateFormula();drawNet();
}

// ---- Input sliders ----
function setupInputSliders(){
  inputs=Array(nIn).fill(0).map((_,i)=>typeof inputs[i]==='number'?inputs[i]:0.5);
  let c=document.getElementById('input-sliders');
  c.innerHTML='';
  for(let i=0;i<nIn;i++){
    (function(idx){
      let d=document.createElement('div');
      d.className='slider-row';
      d.innerHTML=`<label>x${idx+1}</label><input type="range" min="-2" max="2" step="0.1" value="${inputs[idx]}"><span>${parseFloat(inputs[idx]).toFixed(1)}</span>`;
      let sl=d.querySelector('input');
      let sp=d.querySelector('span');
      sl.addEventListener('input',function(){
        inputs[idx]=parseFloat(this.value);
        sp.textContent=parseFloat(this.value).toFixed(1);
        drawNet();updateFormula();
      });
      c.appendChild(d);
    })(i);
  }
}

// ---- Canvas helper ----
function getCtx(id,h){
  let c=document.getElementById(id);
  if(!c)return null;
  let dpr=window.devicePixelRatio||1;
  let w=c.parentElement.clientWidth-40;
  c.style.width=w+'px';
  c.style.height=h+'px';
  c.width=w*dpr;
  c.height=h*dpr;
  let ctx=c.getContext('2d');
  ctx.scale(dpr,dpr);
  return{ctx,w,h};
}

// ---- Draw network ----
function drawNet(){
  let r=getCtx('netCanvas',220);
  if(!r)return;
  let{ctx,w,h}=r;
  ctx.clearRect(0,0,w,h);
  let layerSizes=[nIn,nHid,nOut];
  let nL=3;
  let colW=w/nL;
  let{h:fh,o}=forward(inputs);
  let vals=[inputs,[...fh],[...o]];
  let colors=['#1D9E75','#7F77DD','#D85A30'];

  function neuronY(l,i){
    let n=layerSizes[l];
    let spacing=Math.min(44,(h-48)/Math.max(n-1,1));
    let totalH=(n-1)*spacing;
    return h/2-totalH/2+i*spacing;
  }

  for(let l=0;l<2;l++){
    for(let i=0;i<layerSizes[l];i++){
      for(let j=0;j<layerSizes[l+1];j++){
        let x1=colW*(l+0.5),y1=neuronY(l,i);
        let x2=colW*(l+1.5),y2=neuronY(l+1,j);
        let wt=l===0?W1[j][i]:W2[j<nOut?j:0][i];
        let alpha=Math.min(0.9,Math.abs(wt)*0.5+0.12);
        ctx.strokeStyle=wt>0?`rgba(29,158,117,${alpha})`:`rgba(226,75,74,${alpha})`;
        ctx.lineWidth=Math.min(2.5,Math.abs(wt)*1.2+0.3);
        ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
      }
    }
  }

  let labels=['Input','Hidden','Output'];
  for(let l=0;l<3;l++){
    ctx.fillStyle='#8b8aa0';
    ctx.font='11px sans-serif';
    ctx.textAlign='center';ctx.textBaseline='top';
    ctx.fillText(labels[l],colW*(l+0.5),4);
    for(let i=0;i<layerSizes[l];i++){
      let x=colW*(l+0.5),y=neuronY(l,i);
      let val=vals[l][i]||0;
      let r=18;
      ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle=colors[l]+'22';ctx.fill();
      ctx.strokeStyle=colors[l];ctx.lineWidth=2;ctx.stroke();
      ctx.fillStyle=colors[l];
      ctx.font='bold 11px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(val.toFixed(2),x,y);
    }
  }
}

// ---- Draw activation function ----
function drawActFn(){
  let r=getCtx('actCanvas',120);
  if(!r)return;
  let{ctx,w,h}=r;
  ctx.clearRect(0,0,w,h);
  let pad=24;
  let gw=w-2*pad,gh=h-2*pad;
  ctx.strokeStyle='#2a2a38';ctx.lineWidth=0.8;
  ctx.beginPath();ctx.moveTo(pad,pad+gh/2);ctx.lineTo(pad+gw,pad+gh/2);ctx.stroke();
  ctx.beginPath();ctx.moveTo(pad+gw/2,pad);ctx.lineTo(pad+gw/2,pad+gh);ctx.stroke();
  ctx.strokeStyle='#7F77DD';ctx.lineWidth=2.5;ctx.beginPath();
  for(let px=0;px<gw;px++){
    let x=(px/gw)*6-3;
    let y=activate(x);
    let sy=pad+gh-(y+1.5)/3*gh;
    px===0?ctx.moveTo(pad+px,sy):ctx.lineTo(pad+px,sy);
  }
  ctx.stroke();
  let labels={sigmoid:'σ(x) = 1/(1+e⁻ˣ)',relu:'f(x) = max(0, x)',tanh:'f(x) = tanh(x)',leaky:'f(x) = x>0 ? x : 0.01x'};
  ctx.fillStyle='#8b8aa0';ctx.font='11px sans-serif';ctx.textAlign='left';ctx.textBaseline='top';
  ctx.fillText(labels[actFn],pad+4,pad+2);
}

// ---- Update forward pass formula ----
function updateFormula(){
  let{hz,h,oz,o}=forwardRaw(inputs);
  let lines=[];
  lines.push('<span class="dim">Step 1 — hidden layer (z = Wx + b, then activate):</span>');
  for(let i=0;i<Math.min(nHid,2);i++){
    let terms=inputs.map((v,j)=>`<span class="hl">${W1[i][j].toFixed(2)}</span>×${v.toFixed(1)}`).join(' + ');
    lines.push(`  h${i+1}: z = ${terms} + ${b1[i].toFixed(2)} = <span class="hl">${hz[i].toFixed(3)}</span> → ${actFn}(z) = <span class="hl">${h[i].toFixed(3)}</span>`);
  }
  if(nHid>2)lines.push(`  <span class="dim">... (${nHid-2} more hidden neurons)</span>`);
  lines.push('');
  lines.push('<span class="dim">Step 2 — output layer (sigmoid):</span>');
  for(let i=0;i<nOut;i++){
    let terms=h.map((v,j)=>`<span class="hl">${W2[i][j].toFixed(2)}</span>×${v.toFixed(3)}`).join(' + ');
    let cls=o[i]>0.5?'pos':'neg';
    lines.push(`  y${i+1}: σ(${terms} + ${b2[i].toFixed(2)}) = <span class="${cls}">${o[i].toFixed(4)}</span>  →  prediction: <span class="${cls}">${o[i]>0.5?'1  ✓':'0  ✗'}</span>`);
  }
  document.getElementById('formula-box').innerHTML=lines.join('\n');
}

// ---- Loss curve ----
function drawLoss(){
  let r=getCtx('lossCanvas',180);
  if(!r)return;
  let{ctx,w,h}=r;
  ctx.clearRect(0,0,w,h);
  if(lossHistory.length<2)return;
  let pad=32;let gw=w-pad-8;let gh=h-pad-8;
  let maxL=Math.max(...lossHistory,0.01);
  ctx.strokeStyle='#2a2a38';ctx.lineWidth=0.8;
  ctx.beginPath();ctx.moveTo(pad,8);ctx.lineTo(pad,8+gh);ctx.lineTo(pad+gw,8+gh);ctx.stroke();
  ctx.strokeStyle='#7F77DD';ctx.lineWidth=2;ctx.beginPath();
  lossHistory.forEach((l,i)=>{
    let x=pad+i/Math.max(lossHistory.length-1,1)*gw;
    let y=8+gh-(l/maxL)*gh;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.fillStyle='#8b8aa0';ctx.font='11px sans-serif';
  ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText('Loss',pad+4,10);
  ctx.textAlign='right';ctx.fillText(maxL.toFixed(3),pad-2,8);
  ctx.fillText('0',pad-2,8+gh-8);
  ctx.textAlign='center';ctx.fillText('Epochs →',pad+gw/2,8+gh+10);
}

// ---- Decision boundary ----
function drawDecision(){
  let r=getCtx('decisionCanvas',180);
  if(!r)return;
  let{ctx,w,h}=r;
  ctx.clearRect(0,0,w,h);
  if(nIn!==2)return;
  let pad=20;let gw=w-2*pad;let gh=h-2*pad;
  let res=35;
  for(let gx=0;gx<res;gx++){
    for(let gy=0;gy<res;gy++){
      let ix=(gx/res)*2-1;
      let iy=(gy/res)*2-1;
      let{o}=forward([ix,iy]);
      let v=o[0];
      ctx.fillStyle=v>0.5?`rgba(29,158,117,${0.12+v*0.28})`:`rgba(226,75,74,${0.12+(1-v)*0.28})`;
      ctx.fillRect(pad+gx/res*gw,pad+gy/res*gh,gw/res+1,gh/res+1);
    }
  }
  xorData.forEach(([inp,tgt])=>{
    let px=pad+(inp[0]+1)/2*gw;
    let py=pad+(inp[1]+1)/2*gh;
    ctx.beginPath();ctx.arc(px,py,7,0,Math.PI*2);
    ctx.fillStyle=tgt[0]===1?'#1D9E75':'#E24B4A';
    ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
  });
  ctx.fillStyle='#8b8aa0';ctx.font='11px sans-serif';
  ctx.textAlign='center';ctx.textBaseline='top';
  ctx.fillText('XOR decision boundary',pad+gw/2,2);
}

// ---- XOR table ----
function updateXORTable(){
  let tb=document.getElementById('xor-table');
  if(!tb)return;
  tb.innerHTML='';
  xorData.forEach(([inp,tgt])=>{
    let{o}=forward(inp);
    let pred=Math.round(o[0]);
    let ok=pred===tgt[0];
    let tr=document.createElement('tr');
    tr.innerHTML=`<td>${inp[0]}</td><td>${inp[1]}</td><td>${tgt[0]}</td><td class="${ok?'correct':'wrong'}">${pred}</td><td>${ok?'✓':'✗'}</td>`;
    tb.appendChild(tr);
  });
}

// ---- Init ----
window.addEventListener('load',()=>{
  initNet();
  setupInputSliders();
  setTimeout(()=>{
    drawNet();drawActFn();updateFormula();drawLoss();drawDecision();updateXORTable();
  },60);
});
window.addEventListener('resize',()=>{
  drawNet();drawActFn();drawLoss();drawDecision();
});
