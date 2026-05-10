// Animated floating network for hero section
(function(){
  const canvas=document.getElementById('heroCanvas');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  let W,H,nodes=[],edges=[],t=0;

  function resize(){
    let dpr=window.devicePixelRatio||1;
    W=canvas.parentElement.clientWidth;
    H=320;
    canvas.style.width=W+'px';
    canvas.style.height=H+'px';
    canvas.width=W*dpr;
    canvas.height=H*dpr;
    ctx.scale(dpr,dpr);
    build();
  }

  function build(){
    nodes=[];edges=[];
    let layers=[3,4,4,2];
    let colW=W/(layers.length+1);
    layers.forEach((n,l)=>{
      for(let i=0;i<n;i++){
        let spacing=H/(n+1);
        nodes.push({
          x:colW*(l+1),
          y:spacing*(i+1),
          r:12,
          layer:l,
          idx:i,
          phase:Math.random()*Math.PI*2,
          color:l===0?'#1D9E75':l===layers.length-1?'#D85A30':'#7F77DD'
        });
      }
    });
    for(let a=0;a<nodes.length;a++){
      for(let b=0;b<nodes.length;b++){
        if(nodes[b].layer===nodes[a].layer+1){
          edges.push({a,b,w:(Math.random()-0.5)*2});
        }
      }
    }
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    t+=0.018;

    edges.forEach(e=>{
      let na=nodes[e.a],nb=nodes[e.b];
      let alpha=Math.min(0.55,Math.abs(e.w)*0.4+0.08);
      ctx.strokeStyle=e.w>0?`rgba(29,158,117,${alpha})`:`rgba(226,75,74,${alpha})`;
      ctx.lineWidth=Math.abs(e.w)*1.2+0.3;
      ctx.beginPath();ctx.moveTo(na.x,na.y);ctx.lineTo(nb.x,nb.y);ctx.stroke();
    });

    // Animate signal pulses along edges
    edges.forEach((e,i)=>{
      if((t*0.7+i*0.3)%1.4<0.05){
        let na=nodes[e.a],nb=nodes[e.b];
        let pct=((t*1.2+i*0.17)%1);
        let px=na.x+(nb.x-na.x)*pct;
        let py=na.y+(nb.y-na.y)*pct;
        ctx.beginPath();ctx.arc(px,py,3,0,Math.PI*2);
        ctx.fillStyle=e.w>0?'rgba(29,158,117,0.85)':'rgba(226,75,74,0.85)';
        ctx.fill();
      }
    });

    nodes.forEach(n=>{
      let pulse=Math.sin(t*1.4+n.phase)*0.18+0.82;
      ctx.beginPath();ctx.arc(n.x,n.y,n.r*pulse,0,Math.PI*2);
      ctx.fillStyle=n.color+'22';ctx.fill();
      ctx.strokeStyle=n.color;ctx.lineWidth=1.8;ctx.stroke();
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize',resize);
  resize();
  draw();
})();
