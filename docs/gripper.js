// Original point-and-line artwork derived from the project's pinned R5a meshes.
(() => {
  'use strict';
  const model = window.R5_GRIPPER_MODEL;
  const canvas = document.getElementById('gripper-canvas');
  if (!canvas || !model) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const stage = document.getElementById('gripper-stage');
  const opening = document.getElementById('gripper-opening');
  const rotation = document.getElementById('gripper-motion');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const colors = ['166,156,233', '129,227,228', '201,177,255'];
  const parts = model.parts;
  let width=1, height=1, frame=0, last=0, time=0, visible=false, paused=reduce.matches;
  let yaw=-.46, pitch=-.57, roll=-.35, joint=Number(opening.value)/100*44, drag=null;
  const pointer={x:-1000,y:-1000,active:false};
  // Row-major view matrix; X of the CAD points from the wrist to the fingertips.
  function matrix() {
    const a=yaw+Math.sin(time*.35)*.13, b=pitch, c=roll;
    const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b),cc=Math.cos(c),sc=Math.sin(c);
    return [cc*ca-sc*sb*sa,-sc*cb,cc*sa+sc*sb*ca,sc*ca+cc*sb*sa,cc*cb,sc*sa-cc*sb*ca,-cb*sa,sb,cb*ca];
  }
  function draw() {
    const m=matrix(), scale=Math.min(width/240,height/200)*1.12;
    ctx.clearRect(0,0,width,height);
    const near=[];
    function project(x,y,z,part,normal=false) {
      if(!normal){x+=part.origin[0]+part.axis[0]*joint;y+=part.origin[1]+part.axis[1]*joint;z+=part.origin[2]+part.axis[2]*joint;x-=78;}
      const u=y,v=x,w=z;
      const rx=m[0]*u+m[1]*v+m[2]*w,ry=m[3]*u+m[4]*v+m[5]*w,rz=m[6]*u+m[7]*v+m[8]*w;
      if(normal)return rz;
      const perspective=650/(650-rz);
      return [width*.5+rx*scale*perspective,height*.49+ry*scale*perspective,rz];
    }
    // A quiet reference orbit makes the floating mechanical form legible.
    ctx.strokeStyle='rgba(157,151,210,.10)';ctx.lineWidth=.7;
    ctx.beginPath();ctx.ellipse(width*.5,height*.56,width*.38,height*.29,-.24,0,Math.PI*2);ctx.stroke();
    for(let i=0;i<42;i++){
      const x=((i*191%997)/997)*width,y=((i*317%991)/991)*height;
      ctx.fillStyle=i%7===0?'rgba(167,190,228,.35)':'rgba(167,190,228,.12)';ctx.fillRect(x,y,i%7===0?1.5:1,1);
    }
    parts.forEach((part,pi)=>{
      const {points,edges}=part,color=colors[pi],buckets=Array.from({length:7},()=>[]),stride=width<420?12:6;
      ctx.beginPath();
      for(let i=0;i<edges.length;i+=6){const a=project(edges[i],edges[i+1],edges[i+2],part),b=project(edges[i+3],edges[i+4],edges[i+5],part);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}
      ctx.strokeStyle=`rgba(${color},.23)`;ctx.lineWidth=.55;ctx.stroke();
      for(let i=0;i<points.length;i+=stride){
        const p=project(points[i],points[i+1],points[i+2],part);
        const normal=project(points[i+3],points[i+4],points[i+5],part,true);
        const d=pointer.active?Math.hypot(pointer.x-p[0],pointer.y-p[1]):1000;
        const hot=Math.max(0,1-d/100);
        const shade=Math.min(6,Math.floor((.15+Math.max(0,normal)*.65+hot*.5)*6));
        buckets[shade].push(p[0],p[1],(.63+(p[2]+100)/250)*Math.min(1,width/500+.35)+hot*.45);
        if(hot>.15&&i%54===0)near.push({x:p[0],y:p[1],hot});
      }
      buckets.forEach((bucket,bi)=>{
        ctx.beginPath();for(let j=0;j<bucket.length;j+=3){ctx.moveTo(bucket[j]+bucket[j+2],bucket[j+1]);ctx.arc(bucket[j],bucket[j+1],bucket[j+2],0,Math.PI*2);}
        ctx.fillStyle=`rgba(${color},${.18+bi*.12})`;ctx.fill();
      });
    });
    if(pointer.active){
      const local=near.sort((a,b)=>b.hot-a.hot).slice(0,28);
      ctx.lineWidth=.7;
      for(let i=0;i<local.length;i++){
        const p=local[i];ctx.fillStyle=`rgba(224,239,255,${p.hot*.9})`;ctx.beginPath();ctx.arc(p.x,p.y,1.5,0,Math.PI*2);ctx.fill();
        for(let j=i+1;j<local.length;j++){
          const q=local[j],d=Math.hypot(p.x-q.x,p.y-q.y);
          if(d<60){ctx.strokeStyle=`rgba(161,212,245,${(1-d/60)*p.hot*.48})`;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}
        }
      }
      const glow=ctx.createRadialGradient(pointer.x,pointer.y,0,pointer.x,pointer.y,90);glow.addColorStop(0,'#a7c7ff14');glow.addColorStop(1,'#a7c7ff00');ctx.fillStyle=glow;ctx.fillRect(pointer.x-90,pointer.y-90,180,180);
    }
    stage.classList.add('ready');
  }
  function tick(now){frame=0;if(paused||!visible||document.hidden)return;if(now-last>32){time+=Math.min((now-last)/1000,.08);last=now;draw();}frame=requestAnimationFrame(tick);}
  function update(){cancelAnimationFrame(frame);frame=0;last=performance.now();draw();if(!paused&&visible&&!document.hidden)frame=requestAnimationFrame(tick);}
  function resize(){const r=stage.getBoundingClientRect();width=r.width;height=r.height;const dpr=Math.min(devicePixelRatio||1,1.75);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
  function move(event){
    const r=stage.getBoundingClientRect();pointer.x=event.clientX-r.left;pointer.y=event.clientY-r.top;pointer.active=true;
    if(drag){yaw+=(event.clientX-drag.x)*.008;pitch=Math.max(-1.25,Math.min(1.25,pitch+(event.clientY-drag.y)*.008));drag={x:event.clientX,y:event.clientY};}
    if(paused||drag)draw();
  }
  stage.addEventListener('pointermove',move,{passive:true});
  stage.addEventListener('pointerdown',event=>{if(event.button!==0)return;drag={x:event.clientX,y:event.clientY};stage.setPointerCapture(event.pointerId);stage.classList.add('dragging');});
  function release(){drag=null;stage.classList.remove('dragging');}
  stage.addEventListener('pointerup',release);stage.addEventListener('pointercancel',release);stage.addEventListener('lostpointercapture',release);
  stage.addEventListener('pointerleave',()=>{pointer.active=false;if(paused)draw();});
  stage.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(event.key))return;
    event.preventDefault();if(event.key==='Home'){yaw=-.46;pitch=-.57;}else if(event.key==='ArrowLeft')yaw-=.14;else if(event.key==='ArrowRight')yaw+=.14;else pitch=Math.max(-1.25,Math.min(1.25,pitch+(event.key==='ArrowDown'?.12:-.12)));draw();
  });
  opening.addEventListener('input',()=>{joint=Number(opening.value)/100*44;document.getElementById('gripper-opening-value').value=opening.value+'%';draw();});
  document.getElementById('gripper-reset').addEventListener('click',()=>{yaw=-.46;pitch=-.57;time=0;draw();});
  rotation.addEventListener('click',()=>document.getElementById('motion-toggle').click());
  new ResizeObserver(resize).observe(stage);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;update();},{threshold:.05}).observe(stage);
  document.addEventListener('visibilitychange',update);
  window.LLM4RobotGripper={setPaused(value){paused=Boolean(value);rotation.setAttribute('aria-pressed',String(paused));rotation.textContent=paused?'Resume motion ▷':'Pause motion Ⅱ';update();}};
  resize();update();
})();
