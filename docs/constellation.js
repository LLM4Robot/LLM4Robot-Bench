(() => {
  'use strict';
  const canvas = document.getElementById('constellation');
  if (!canvas) return;
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) return;
  const hero = canvas.parentElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let width = 0, height = 0, dpr = 1, frame = 0, clock = 0, last = 0;
  let paused = reduced.matches, visible = true;
  const mouse = { x: -1000, y: -1000, tx: -1000, ty: -1000, active: false };
  let seed = 1847;
  const random = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
  const stars = Array.from({length: 150}, () => ({x:random(), y:random(), r:random()*.9+.25, a:random()*.35+.08}));
  const particles = Array.from({length: 1450}, (_, i) => {
    const angle = random() * Math.PI * 2;
    const lane = i % 3;
    const band = random() * random();
    const radius = .46 + band * .30;
    return {angle, radius, y:(random()-.5)*.15, z:(random()-.5)*.22, phase:random()*6.28, size:random()*1.15+.3, lane, hot:i%21===0};
  });
  function resize() {
    const rect = hero.getBoundingClientRect();
    width = rect.width; height = rect.height; dpr = Math.min(devicePixelRatio || 1, 1.75);
    canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw();
  }
  function draw() {
    ctx.fillStyle = '#080a10'; ctx.fillRect(0,0,width,height);
    const scale = Math.min(width*.79,920), cy=height*.47;
    const glow = ctx.createRadialGradient(width*.57,cy+80,0,width*.57,cy+80,scale*.7);
    glow.addColorStop(0,'#23244535'); glow.addColorStop(.5,'#2020431b'); glow.addColorStop(1,'#080a1000');
    ctx.fillStyle=glow;ctx.fillRect(0,0,width,height);
    stars.forEach(s=> {ctx.globalAlpha=s.a*(.85+.15*Math.sin(clock*.25+s.x*30));ctx.fillStyle='#d4d8ef';ctx.beginPath();ctx.arc(s.x*width,s.y*height,s.r,0,Math.PI*2);ctx.fill();});
    ctx.globalAlpha=1;
    const projected=[];
    const count = width<600 ? 800 : particles.length;
    for(let i=0;i<count;i++) {
      const p=particles[i];const a=p.angle+clock*.025;
      const twist=Math.sin(a*2+clock*.08)*.10;
      let x=Math.cos(a)*p.radius;
      let y=Math.sin(a)*p.radius*.37 + p.y + twist;
      const z=p.z+Math.sin(a)*.18;
      const slant=-.19;
      const rx=x*Math.cos(slant)-y*Math.sin(slant);
      const ry=x*Math.sin(slant)+y*Math.cos(slant);
      let px=width*.5+rx*scale*(1+z*.3),py=cy+ry*scale;
      const dx=px-mouse.x,dy=py-mouse.y,dist=Math.hypot(dx,dy);
      const proximity=mouse.active ? Math.max(0,1-dist/180) : 0;
      if(dist>1){px+=dx/dist*proximity*13;py+=dy/dist*proximity*13;}
      const aOpacity=(.22+(z+.3)*.68)*(p.hot?1.5:1)+proximity*.35;
      const color=p.lane===0?'175,154,239':p.lane===1?'121,183,209':'195,191,224';
      ctx.fillStyle=`rgba(${color},${Math.min(aOpacity,.9)})`;
      ctx.beginPath();ctx.arc(px,py,p.size*(1+proximity*.7),0,Math.PI*2);ctx.fill();
      if(p.hot){projected.push({x:px,y:py,near:proximity});ctx.fillStyle=`rgba(223,222,255,${.30+proximity*.6})`;ctx.fillRect(px-2,py-.35,4,.7);ctx.fillRect(px-.35,py-2,.7,4);}
    }
    if(mouse.active){
      const close=projected.filter(p=>p.near>0).slice(0,15);
      for(let i=0;i<close.length;i++){
        const p=close[i];
        ctx.strokeStyle=`rgba(176,182,242,${p.near*.3})`;ctx.lineWidth=.7;
        ctx.beginPath();ctx.moveTo(mouse.x,mouse.y);ctx.lineTo(p.x,p.y);ctx.stroke();
        for(let j=i+1;j<close.length;j++){
          const q=close[j],d=Math.hypot(p.x-q.x,p.y-q.y);
          if(d<105){ctx.strokeStyle=`rgba(156,192,236,${(1-d/105)*.4})`;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}
        }
      }
      const halo=ctx.createRadialGradient(mouse.x,mouse.y,0,mouse.x,mouse.y,130);
      halo.addColorStop(0,'#b1a2ff10');halo.addColorStop(1,'#b1a2ff00');ctx.fillStyle=halo;ctx.fillRect(mouse.x-130,mouse.y-130,260,260);
    }
  }
  function tick(now) {
    frame=0;
    if(paused||!visible||document.hidden)return;
    if(now-last>28){clock+=Math.min((now-last)/1000,.06);last=now;mouse.x+=(mouse.tx-mouse.x)*.15;mouse.y+=(mouse.ty-mouse.y)*.15;draw();}
    frame=requestAnimationFrame(tick);
  }
  function update(){cancelAnimationFrame(frame);frame=0;last=performance.now();if(!paused&&visible&&!document.hidden)frame=requestAnimationFrame(tick);else draw();}
  hero.addEventListener('pointermove',e=>{
    if(e.pointerType==='touch')return;
    const r=hero.getBoundingClientRect();mouse.tx=e.clientX-r.left;mouse.ty=e.clientY-r.top;
    if(!mouse.active){mouse.x=mouse.tx;mouse.y=mouse.ty;}mouse.active=true;
    if(paused){mouse.x=mouse.tx;mouse.y=mouse.ty;draw();}
  },{passive:true});
  hero.addEventListener('pointerleave',()=>{mouse.active=false;if(paused)draw();});
  new ResizeObserver(resize).observe(hero);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;update();},{threshold:0}).observe(hero);
  document.addEventListener('visibilitychange',update);
  window.LLM4RobotSky={setPaused(value){paused=Boolean(value);update();}};
  resize();update();
})();
