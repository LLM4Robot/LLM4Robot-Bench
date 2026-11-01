(() => {
  'use strict';
  const tasks=window.BENCHMARK_TASKS,films=window.BENCHMARK_FILMS;
  const $=id=>document.getElementById(id);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const nav=document.querySelector('.nav-shell');
  const updateNav=()=>nav.classList.toggle('scrolled',scrollY>30);
  addEventListener('scroll',updateNav,{passive:true});updateNav();
  const titles={'L1-01':'Marker Placement','L1-02':'Charger alignment','L1-03':'Hat hanging','L1-04':'Cup nesting','L1-05':'Flower transfer','L2-01':'Marker Cap Removal','L2-02':'Book pulling','L2-03':'Marble pouring','L2-04':'Orange handling','L2-05':'Drawer pull & return','L2-06':'Jelly handling','L2-07':'Whiteboard wiping','L3-01':'Bottle Cap Removal','L3-02':'Nut Removal','L3-03':'Stacked-book pushing','L3-04':'Block sweeping'};
  let selected=tasks.find(t=>t.id==='L2-02'),view='overview',setting='original',level='all';
  let filmIndex=-1,filmVisible=false,filmPaused=false,motionPaused=reduced.matches,manualFilmPlay=false,progressFrame=0;
  const videos=[$('film-sim'),$('film-real')],reel=$('film-reel');
  const allVideos=[...videos,reel];
  const compactReel=matchMedia('(max-width:760px)');
  const demoMedia=(name,ext)=>`media/demos/${name}.${ext}?v=${window.BENCHMARK_DEMOS?.[name]?.sha256?.slice(0,12)||'1'}`;
  const filmMedia=(name,ext)=>`media/loops/${name}.${ext}?v=02f0d6b171ab`;
  function loadReel(){
    const name=compactReel.matches?'overview-mobile':'overview';
    if(reel.getAttribute('src')===filmMedia(name,'mp4'))return;
    reel.poster=filmMedia(name,'webp');reel.src=filmMedia(name,'mp4');reel.load();
  }
  compactReel.addEventListener('change',()=>{loadReel();syncPlayback();});
  let toastTimer;
  function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),2200);}
  async function copy(text){
    try{
      if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);
      else{const input=document.createElement('textarea');input.value=text;input.style.cssText='position:fixed;opacity:0';document.body.append(input);input.select();const copied=document.execCommand('copy');input.remove();if(!copied)throw Error('Copy unavailable');}
      toast('Copied to clipboard');
    }catch(_){toast('Select the command to copy it.');}
  }
  function renderList(){
    const list=$('task-list'),scrollLeft=list.scrollLeft,scrollTop=list.scrollTop;list.replaceChildren();
    tasks.filter(t=>level==='all'||t.level===level).forEach(task=>{
      const button=document.createElement('button');button.className='task-button'+(task.id===selected.id?' active':'');button.setAttribute('aria-pressed',String(task.id===selected.id));
      const id=document.createElement('small'),title=document.createElement('span');id.textContent=task.id;title.textContent=titles[task.id]||task.title;button.append(id,title);
      button.addEventListener('click',()=>{selected=task;render();renderList();});$('task-list').append(button);
    });
    list.scrollLeft=scrollLeft;list.scrollTop=scrollTop;
  }
  function render(){
    $('task-id').textContent=selected.id;$('task-title').textContent=titles[selected.id]||selected.title;
    for(const state of ['original','variant01']){
      const image=$(state==='original'?'original-image':'variant-image');image.src=selected.settings[state].images[view];image.alt=`${selected.id} ${titles[selected.id]||selected.title}: ${state} simulation initial state, ${view.replace('_',' ')}`;
    }
    const config=selected.settings[setting];$('task-prompt').textContent=config.instruction;$('task-command').textContent=`python tools/reproduce.py --tasks ${config.id}`;
    $('budget').textContent=`Seed ${config.seed} · ${config.max_calls||'Uncapped'} calls · ${config.wall_time_s?config.wall_time_s/3600+' h':'No time cap'}`;
    $('config-link').href=`https://github.com/LLM4Robot/LLM4Robot-Bench/blob/main/${config.config}`;$('comparison').style.aspectRatio=view==='overview'?'1100/820':'4/3';
  }
  document.querySelectorAll('[data-level]').forEach(button=>button.addEventListener('click',()=>{
    level=button.dataset.level;document.querySelectorAll('[data-level]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    if(level!=='all'&&selected.level!==level)selected=tasks.find(t=>t.level===level);renderList();render();
  }));
  for(const key of ['view','setting'])document.querySelectorAll(`[data-${key}]`).forEach(button=>button.addEventListener('click',()=>{
    if(key==='view')view=button.dataset.view;else setting=button.dataset.setting;
    document.querySelectorAll(`[data-${key}]`).forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});render();
  }));
  $('comparison-slider').addEventListener('input',event=>$('comparison').style.setProperty('--split',`${event.target.value}%`));
  $('copy-command').addEventListener('click',()=>copy($('task-command').textContent));$('copy-setup').addEventListener('click',()=>copy($('setup-command').textContent));
  document.querySelectorAll('[data-explore-level]').forEach(link=>link.addEventListener('click',()=>document.querySelector(`[data-level="${link.dataset.exploreLevel}"]`).click()));
  document.querySelectorAll('.level-card').forEach(card=>card.addEventListener('pointermove',event=>{const r=card.getBoundingClientRect();card.style.setProperty('--mouse-x',`${event.clientX-r.left}px`);card.style.setProperty('--mouse-y',`${event.clientY-r.top}px`);},{passive:true}));
  function progress(){
    const v=filmIndex<0?reel:videos[0];if(v.duration)$('cinema').querySelector('.film-progress>span').style.width=`${(v.currentTime/v.duration)*100}%`;
    if(!v.paused&&!document.hidden)progressFrame=requestAnimationFrame(progress);
  }
  function syncPlayback(){
    const allowed=filmVisible&&!filmPaused&&(!motionPaused||manualFilmPlay)&&!document.hidden;
    allVideos.forEach(v=>{const active=filmIndex<0?v===reel:v!==reel;if(allowed&&active&&!v.ended){v.play().catch(error=>{if(error.name==='AbortError')return;if(filmVisible&&!filmPaused&&(!motionPaused||manualFilmPlay)){filmPaused=true;syncPlayback();}});}else v.pause();});
    const stopped=filmPaused||(motionPaused&&!manualFilmPlay);
    const ended=filmIndex>=0&&videos.every(v=>v.ended);
    $('film-play').innerHTML=ended?'Replay clips <span>↻</span>':stopped?'Play clips <span>▷</span>':'Pause clips <span>Ⅱ</span>';$('film-play').setAttribute('aria-pressed',String(stopped));
    cancelAnimationFrame(progressFrame);if(allowed)progressFrame=requestAnimationFrame(progress);
  }
  function selectFilm(index){
    allVideos.forEach(v=>v.pause());filmIndex=index;const overview=index<0;
    $('reel-panel').hidden=!overview;$('film-pair').hidden=overview;$('gif-download').hidden=!overview;
    if(overview){
      $('film-id').textContent='ALL 16 TASKS';$('film-title').textContent='Sixteen tasks. Two worlds.';$('film-description').textContent='Simulation left. Real robot right. Select a task to explore.';
      loadReel();
    }else{
      const film=films[index];$('film-id').textContent=film.id;$('film-title').textContent=film.title;$('film-description').textContent=film.line;
      ['sim','real'].forEach((domain,i)=>{
        const name=`${film.id}-${domain}`,meta=window.BENCHMARK_DEMOS[name],result=meta.outcome,video=videos[i];
        const badge=$(domain+'-outcome');badge.textContent=result;badge.classList.toggle('failure',result==='Failure');
        video.poster=demoMedia(name,'webp');video.src=demoMedia(name,'mp4');video.load();
        video.setAttribute('aria-label',`${film.title}, ${domain==='sim'?'simulation':'real robot'}. Three camera views. Selected trial: ${result}. Edited and sped up for presentation.`);
      });


    }
    document.querySelectorAll('[data-film]').forEach(button=>{const active=Number(button.dataset.film)===index;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
    $('cinema').querySelector('.film-progress>span').style.width='0%';syncPlayback();
  }
  const overviewTab=document.createElement('button');overviewTab.textContent='Overview';overviewTab.dataset.film='-1';overviewTab.addEventListener('click',()=>selectFilm(-1));$('film-tabs').append(overviewTab);
  films.forEach((film,index)=>{
    const cell=document.createElement('button');cell.dataset.film=index;cell.className='reel-cell';cell.setAttribute('aria-label',`Explore ${film.id}: ${film.title}, simulation left and real robot right`);cell.innerHTML=`<span>${film.title} <i>↗</i></span><small>${film.id} · SIM / REAL</small>`;cell.addEventListener('click',()=>{selectFilm(index);$('film-tabs').querySelector(`[data-film="${index}"]`).focus({preventScroll:true});if(compactReel.matches)$('cinema').scrollIntoView({behavior:motionPaused?'instant':'smooth',block:'start'});});$('reel-labels').append(cell);
    const tab=document.createElement('button');tab.textContent=film.short;tab.dataset.film=index;tab.addEventListener('click',()=>selectFilm(index));$('film-tabs').append(tab);
    const thumb=document.createElement('button');thumb.className='film-thumb';thumb.dataset.film=index;thumb.setAttribute('aria-label',`Watch ${film.title}`);
    thumb.innerHTML=`<div class="thumb-image"><img src="${filmMedia(film.id+'-sim','webp')}" alt="${film.title} simulation excerpt" loading="lazy" width="400" height="260"><span class="thumb-play" aria-hidden="true">▷</span></div><span><small>${film.id}</small>${film.title}</span>`;
    thumb.addEventListener('click',()=>{selectFilm(index);if(!matchMedia('(min-width:761px)').matches)$('cinema').scrollIntoView({behavior:motionPaused?'instant':'smooth',block:'center'});});$('film-strip').append(thumb);
  });
  function applyMotion(){
    document.body.classList.toggle('motion-paused',motionPaused);$('motion-toggle').setAttribute('aria-pressed',String(motionPaused));$('motion-toggle').querySelector('.motion-label').textContent=motionPaused?'Resume motion':'Pause motion';$('motion-toggle').querySelector('.motion-icon').textContent=motionPaused?'▷':'Ⅱ';window.LLM4RobotSky?.setPaused(motionPaused);window.LLM4RobotGripper?.setPaused(motionPaused);syncPlayback();
  }
  $('motion-toggle').addEventListener('click',()=>{motionPaused=!motionPaused;manualFilmPlay=false;applyMotion();});
  videos.forEach(v=>v.addEventListener('ended',()=>{if(videos.every(x=>x.ended))syncPlayback();}));
  $('film-play').addEventListener('click',()=>{if(filmIndex>=0&&videos.every(v=>v.ended)){videos.forEach(v=>v.currentTime=0);filmPaused=false;manualFilmPlay=true;syncPlayback();return;}if(motionPaused&&!manualFilmPlay){manualFilmPlay=true;filmPaused=false;}else{filmPaused=!filmPaused;}syncPlayback();});
  reduced.addEventListener('change',event=>{motionPaused=event.matches;manualFilmPlay=false;applyMotion();});
  new IntersectionObserver(entries=>{filmVisible=entries[0].isIntersecting;syncPlayback();},{threshold:.12}).observe($('cinema'));
  document.addEventListener('visibilitychange',syncPlayback);
  allVideos.forEach(v=>v.addEventListener('error',()=>{filmPaused=true;syncPlayback();toast('Video could not load. Try another experiment.');}));
  renderList();render();selectFilm(filmIndex);applyMotion();
})();
