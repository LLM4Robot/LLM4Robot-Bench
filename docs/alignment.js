// Paired evidence from paper Figure 5. See media/alignment/sources.json.
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const chapters = {
    workspace: {
      title: 'Start from comparable conditions.',
      description: 'Measured tabletop geometry, object placement and support conditions establish the task before the robot moves.',
      metric: ['1:1', 'TASK PAIRING'],
      views: [
        {name:'Layout',image:'workspace',details:['Reconstructed workspace','Physical workspace'],note:'Paper Figure 5(a) · Block sweeping workspace and initial object placement.'},
        {name:'Nut engagement',image:'engagement',details:['Simulated screw–nut assembly','Physical screw–nut assembly'],note:'Paper Figure 5(a) · Matching the initial screw–nut engagement sets the rotation remaining before separation.'}
      ]
    },
    robot: {
      title: 'The same embodiment, in both worlds.',
      description: 'The open-source ARX R5 model reproduces the arm geometry and kinematic structure. Arm mounting, two-finger grippers and their relationship to the table follow the physical setup.',
      metric: ['R5', 'ARX ROBOTICS'],
      views: [
        {name:'Arm bases',image:'robot',details:['Simulated dual-arm mounting','Physical dual-arm mounting'],note:'Paper Figure 5(c) · Two ARX R5 arms, each with six revolute joints and an R5-series two-finger gripper.'},
        {name:'Grippers',image:'gripper',details:['Simulated two-finger gripper','Physical two-finger gripper'],note:'Paper Figure 5(c) · Empty grippers expose the finger geometry and available contact surfaces.'}
      ]
    },
    cameras: {
      title: 'Match what the robot can see.',
      description: 'One external view and two wrist views capture workspace context and local contacts. Real images guide viewpoint, image-scale and visibility matching.',
      metric: ['3', 'CAMERA VIEWS'],
      views: [
        {name:'External',image:'overview',details:['Simulated external view','Real external view'],note:'Paper Figure 5(d) · Initial-state external views; real wrist frames selected at 16:40.5 and 28:18 for comparable geometry. These views are not time-synchronized. Camera matching is image-guided.'},
        {name:'Left wrist',image:'left',details:['Simulated left-wrist view','Real left wrist · 16:40.5'],note:'Paper Figure 5(d) · Simulated initial state paired with a real execution frame at 16:40.5 for comparable gripper–box geometry. Not time-synchronized.'},
        {name:'Right wrist',image:'right',details:['Simulated right-wrist view','Real right wrist · 28:18'],note:'Paper Figure 5(d) · Real execution frame at 28:18 selected for comparable gripper–dustpan geometry. Not time-synchronized.'}
      ]
    },
    geometry: {
      title: 'Real measurements set the scale.',
      description: 'Multi-view images guide shape and surface appearance. Measured dimensions set physical scale, including the capped marker’s 150 mm length.',
      metric: ['150', 'MILLIMETRES'],
      views: [{name:'Marker length',image:'geometry',details:['Configured length · 150 mm','Ruler measurement · 150 mm'],note:'Paper Figure 5(b1) · The ruler photograph documents the real marker length. The views are shown at different image scales.'}]
    },
    mass: {
      title: 'Give the objects physical weight.',
      description: 'Measured mass anchors the simulation. Friction is refined against observed sliding, and deformable material parameters against physical response.',
      metric: ['26.4', 'GRAMS'],
      views: [{name:'Jelly mass',image:'mass',details:['Configured mass · 26.4 g','Scale reading · 26.4 g'],note:'Paper Figure 5(b2) · Rendered jelly asset and the documented weighing measurement. These images demonstrate mass matching.'}]
    }
  };
  let chapter = 'workspace';
  function render(index = 0) {
    const data = chapters[chapter], view = data.views[index];
    $('match-title').textContent = data.title;
    $('match-description').textContent = data.description;
    $('match-metric').querySelector('strong').textContent = data.metric[0];
    $('match-metric').querySelector('span').textContent = data.metric[1];
    $('match-note').textContent = view.note;
    for (const [i,domain] of ['sim','real'].entries()) {
      $('match-'+domain).src = `media/alignment/${view.image}-${domain}.webp`;
      $('match-'+domain).alt = `${view.details[i]}. ${data.title}`;
      $('match-'+domain+'-detail').textContent = view.details[i];
      const figure = $('match-'+domain).closest('figure');
      let wrists = figure.querySelector('.match-wrists');
      if (!wrists) {
        wrists = document.createElement('div');
        wrists.className = 'match-wrists';
        figure.querySelector('.alignment-image').after(wrists);
      }
      wrists.hidden = chapter !== 'cameras';
      wrists.replaceChildren();
      if (chapter === 'cameras') {
        data.views.slice(1).forEach(camera => {
          const preview = document.createElement('div');
          preview.className = 'match-wrist';
          const image = document.createElement('img');
          image.src = `media/alignment/${camera.image}-${domain}.webp`;
          image.alt = camera.details[i];
          const label = document.createElement('span');
          label.textContent = camera.name;
          preview.append(image,label);
          wrists.append(preview);
        });
      }
    }
    const options = $('match-options');
    options.replaceChildren();
    options.hidden = chapter === 'cameras';
    if (options.hidden) return;
    data.views.forEach((item,i) => {
      const button = document.createElement('button');
      button.textContent = item.name;
      button.classList.toggle('active', i === index);
      button.setAttribute('aria-pressed',String(i === index));
      button.addEventListener('click',() => {render(i);$('match-options').children[i].focus({preventScroll:true});});
      options.append(button);
    });
  }
  document.querySelectorAll('[data-match]').forEach(button => button.addEventListener('click',() => {
    chapter = button.dataset.match;
    document.querySelectorAll('[data-match]').forEach(b => {
      b.classList.toggle('active',b === button);
      b.setAttribute('aria-pressed',String(b === button));
    });
    render();
  }));
  render();
})();
