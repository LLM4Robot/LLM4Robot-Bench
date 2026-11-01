"""Native collider containment, release and load-bearing criteria for L1."""

import numpy as np

from ...robots.transforms import pose_transform
from .evaluation import held, relative, released
from .native_geometry import (band_opening, cavity_fit, clip_below, cup_cavity,
                              shapes, transformed_poly)


def stable_outcome(e, key, rows, active, *, relative_frame=None):
    """One-second outcome stability ignores yaw/spin but measures position/tilt.

    Contact/release/containment must hold at every sample. Bounded drift from
    the interval's first pose prevents slow translation or precession passing.
    """
    if not hasattr(e,'outcome_anchors'): e.outcome_anchors={}
    if not active:
        e.outcome_anchors.pop(key,None)
        return False
    states=[]
    for row in rows:
        transform=relative(row,relative_frame) if relative_frame is not None else pose_transform(row['pose'])
        axis=pose_transform(row['pose'])[:3,2]
        transverse=np.asarray(row['angular_velocity'])-axis*np.dot(row['angular_velocity'],axis)
        if np.linalg.norm(row['linear_velocity']) >= .025 or np.linalg.norm(transverse) >= .3:
            e.outcome_anchors.pop(key,None)
            return False
        states.append((transform[:3,3],transform[:3,2]))
    previous=e.outcome_anchors.get(key)
    if previous is not None and any(np.linalg.norm(pos-old_pos) > .001 or
                                    np.dot(axis,old_axis) < np.cos(.02)
                                    for (pos,axis),(old_pos,old_axis) in zip(states,previous)):
        e.outcome_anchors.pop(key,None)
        return False
    if previous is None:
        e.outcome_anchors[key]=[(pos.copy(),axis.copy()) for pos,axis in states]
    return True


def marker(e, rows):
    """L1 entry-only outcome; release/support/motion are diagnostic evidence."""
    r,p=rows['marker'],e.spec.parameters
    marker_name=p.get('geometry_prefix','')+'whiteboard_marker'
    t=pose_transform(r['pose'])
    cavity=np.linalg.inv(pose_transform(e.spec.design.fixture_pose)) @ t
    inside,margin=cavity_fit(marker_name,cavity,.001,
                            receptacle=e.spec.design.fixture_asset.split(':',1)[1])
    vertices=np.concatenate([transformed_poly(marker_name,i,t)[0]
                             for i in range(len(shapes(marker_name)))])
    depth=p['rim']-vertices[:,2].min()
    support=r['fixture_force_n'] > p['min_support_n'] and r['bench_force_n'] < .05
    if vertices[:,2].min() > p['table_z']+.005 and r['bench_force_n'] < .05:
        e.flags.add('marker_left_table')
    body_on_table=abs(t[2,2]) < .5 and vertices[:,2].min() <= p['table_z']+.003
    if (not held(r) and r['bench_force_n']>.05 and
            ('marker_left_table' in e.flags or body_on_table)):
        e.flags.add('marker_dropped_on_table')
    returned_to_table = 'marker_dropped_on_table' in e.flags
    require_release = p.get('require_marker_release', False)
    complete = inside and (not require_release or (released(r) and support and 'marker' in e.grasped))
    return complete, {'inside_holder':inside,'depth_m':float(depth),'fixture_supported':bool(support),
                  'release_required':require_release,
                  'collider_clearance_m':float(margin),'geometry':'native barrel cylinder + cap capsule',
                  'released':released(r),'dropped_on_table':returned_to_table,
                  'stability':('no additional dwell after supported release' if require_release
                               else 'not required for L1 entry-only success'),
                  'quality_warnings':(['Marker returned to or toppled on the tabletop; recovery allowed']
                                      if returned_to_table else [])}, ''


def plug(e, rows):
    r,p=rows['plug'],e.spec.parameters
    charger_name=p.get('geometry_prefix','')+'usb_charger'
    # Evaluate in the strip frame: rotating its layout must rotate both slot
    # centres and their rectangular cross-sections, not just the visual asset.
    fixture=pose_transform(e.spec.design.fixture_pose)
    t=np.linalg.inv(fixture) @ pose_transform(r['pose']); tol=p['contact_tolerance_m']
    rim=p['rim']-fixture[2,3]; floor=p['floor']-fixture[2,3]
    slots=np.asarray(p['slot_centres'])
    slot_half=np.asarray(p['opening'])/2
    blades=[]
    for i,s in enumerate(shapes(charger_name)):
        if not s['name'].startswith('Blade'):continue
        vertices,edges,_=transformed_poly(charger_name,i,t)
        clipped=clip_below(vertices,edges,rim)
        blades.append((vertices,clipped))
    if p.get('alignment_only'):
        candidates = []
        for order in (slots, slots[::-1]):
            aligned = []
            for (vertices, _), centre in zip(blades, order):
                tip = vertices[np.isclose(vertices[:, 2], vertices[:, 2].min(), atol=0.0001)]
                gap = float(tip[:, 2].min() - rim)
                aligned.append(bool(np.all(np.abs(vertices[:, :2] - centre) <= slot_half + tol)
                                    and -tol <= gap <= p['alignment_max_gap_m']))
            candidates.append(aligned)
        aligned = any(all(candidate) for candidate in candidates)
        done = aligned and t[2, 2] > 0 and held(r) and 'plug' in e.grasped
        return bool(done), {'both_prongs_aligned': aligned, 'insertion_required': False,
                           'release_required': False, 'grasp_lift_recorded': 'plug' in e.grasped,
                           'alignment_max_gap_m': p['alignment_max_gap_m'],
                           'target': p.get('target_socket','native compatible middle blue two-slot pair'),
                           'rubric_status': 'engineering alignment proxy; source tolerance unspecified'}, ''
    assignments=[]
    for order in (slots,slots[::-1]):
        checks=[]
        for (vertices,clipped),centre in zip(blades,order):
            depth=rim-vertices[:,2].min()
            fit=bool(len(clipped) and np.all(np.abs(clipped[:,:2]-centre)<=slot_half+tol)
                     and vertices[:,2].min()>=floor-tol and depth>tol)
            checks.append({'inside_slot':fit,'depth_m':float(depth)})
        assignments.append(checks)
    checks=max(assignments,key=lambda a:sum(c['inside_slot'] for c in a))
    housing_index=next(i for i,s in enumerate(shapes(charger_name)) if s['name']=='Housing')
    housing=transformed_poly(charger_name,housing_index,t)[0]
    # Seating is diagnostic only. Keep a geometric nonpenetration check on the
    # housing so reversed/upside-down insertion cannot put plastic into the strip.
    local=transformed_poly(charger_name,housing_index,np.eye(4))[0]
    shoulder=housing[np.isclose(local[:,2],local[:,2].min(),atol=1e-8)]
    gaps=shoulder[:,2]-rim
    seated=bool(t[2,2]>0 and gaps.min()>=-tol and gaps.max()<=p['seat_gap_m'])
    housing_clear=bool(housing[:,2].min()>=rim-tol)
    inside=bool(all(c['inside_slot'] for c in checks))
    support=r['fixture_force_n']>.05 and r['bench_force_n']<.05
    done=inside and housing_clear
    return done, {'inside_socket':inside,'blades':checks,'housing_seated':seated,
                  'housing_seat_gap_range_m':[float(gaps.min()),float(gaps.max())],
                  'depth_m':min(c['depth_m'] for c in checks),'release_required':False,
                  'entry_epsilon_m':tol,'housing_clear_of_strip':housing_clear,
                  'released':released(r),'fixture_supported':support,
                  'stability':'not required for L1 entry-only success',
                  'grasp_lift_recorded':'plug' in e.grasped,'native_dimensions_retained':True}, ''


def hat(e, rows):
    r,p=rows['hat'],e.spec.parameters
    t=pose_transform(r['pose'])
    a=(np.asarray(p['hook_start'])-t[:3,3]) @ t[:3,:3]
    b=(np.asarray(p['hook_end'])-t[:3,3]) @ t[:3,:3]
    direction=(b-a)/np.linalg.norm(b-a); z=p['band_plane_z']; radius=p['hook_radius']
    through=False
    if abs(direction[2]) > 1e-6:
        f=(z-a[2])/(b[2]-a[2]); hit=a+f*(b-a)
        planes=band_opening(p.get('geometry_prefix','')+'yellow_visor'); normals=planes[:,:2]
        extent=radius*np.sqrt(np.sum(normals**2,axis=1)+(normals @ direction[:2]/direction[2])**2)
        # Both spherical caps must be past the opening plane: balancing the
        # rear band on the rounded end is insufficient, even with contact.
        caps_clear=min(a[2],b[2])+radius<z<max(a[2],b[2])-radius
        through=bool(0<f<1 and caps_clear and np.all(normals@hit[:2]+extent<=planes[:,2]+.001))
    support=(r.get('target_hook_vertical_force_n',0)>p['min_hook_support_n']
             and r['bench_force_n']<.05 and np.min(np.asarray(r['corners'])[:,2])>p['table_z']+.03)
    done=through and support and released(r)
    return done, {'hook_through_head_opening':through,'target_hook_supported':bool(support),
                  'target_hook_vertical_force_n':r.get('target_hook_vertical_force_n',0),
                  'release_required':True,'stability_wait_s':0.,'geometry':'native 64-sector band and rounded Hook0'}, ''


def cups(e, rows):
    top,base=rows['upper_cup'],rows['lower_cup']; p=e.spec.parameters
    cup_name=p.get('geometry_prefix','')+'ripple_paper_cup'
    t=relative(top,base)
    contained,margin=cavity_fit(cup_name,t,p['cup_contact_tolerance'],receptacle=cup_name)
    floor_index=next(i for i,s in enumerate(shapes(cup_name)) if s['name']=='Floor')
    bottom=transformed_poly(cup_name,floor_index,t)[0]
    contained=bool(contained and t[2,2]>0 and pose_transform(base['pose'])[2,2]>0
                   and bottom[:,2].max()<cup_cavity(cup_name)[2]-p['cup_contact_tolerance'])
    supported=(top['pair_forces'].get('lower_cup',0)>p['min_pair_support_n']
               and base['bench_force_n']>p['min_bench_support_n'])
    active=contained and supported and released(top) and released(base)
    if 'centre_xy' in p:
        if held(base, p['lower_cup_arm']):
            e.flags.add('left_cup_grasped_by_assigned_arm')
        centred = np.linalg.norm(np.asarray(base['pose'][:2]) - p['centre_xy']) <= p['centre_tolerance_m']
        if centred and released(base) and base['bench_force_n'] > .05 and 'left_cup_grasped_by_assigned_arm' in e.flags:
            e.flags.add('left_cup_placed_at_centre')
        if held(top, p['upper_cup_arm']) and 'left_cup_placed_at_centre' in e.flags:
            e.flags.add('right_cup_grasped_after_centre_placement')
        active = active and centred and 'right_cup_grasped_after_centre_placement' in e.flags
    # Relative nesting geometry and the base's world pose must both stay put.
    pair_stable=stable_outcome(e,'cup_pair',[top],active,relative_frame=base)
    base_stable=stable_outcome(e,'cup_base',[base],active)
    return pair_stable and base_stable, {'nested':contained,'cup_on_cup_support':bool(supported),
                 'relative_bottom_height_m':float(t[2,3]),'collider_clearance_m':float(margin),
                 'geometry':'native segmented cup walls and floor',
                 'stability':'relative position and tilt; axial spin ignored'}, ''


def flower(e, rows):
    r,p=rows['rose'],e.spec.parameters; t=pose_transform(r['pose'])
    rose_name=p.get('geometry_prefix','')+'artificial_rose'
    cup_name=p.get('geometry_prefix','')+'ripple_paper_cup'
    cavity=(relative(r,rows['target_cup']) if p.get('dynamic_target_cup') else t.copy())
    if not p.get('dynamic_target_cup'):
        cavity[:2,3]-=p['vase_xy']; cavity[2,3]-=p['table_z']
    stems=[i for i,s in enumerate(shapes(rose_name)) if 'stem' in s['name'].lower()]
    # Check all immersed stem segments, including bends. The bud's complete
    # collision envelope must remain above the mouth.
    inside,margin=cavity_fit(rose_name,cavity,.001,indices=stems,receptacle=cup_name)
    geometry_frame = cavity if p.get('dynamic_target_cup') else t
    _, cavity_floor, cavity_rim = cup_cavity(cup_name)
    floor, rim = ((cavity_floor, cavity_rim) if p.get('dynamic_target_cup') else (p['floor'], p['rim']))
    tip=transformed_poly(rose_name,stems[0],geometry_frame)[0]
    inside=bool(inside and tip[:,2].min()<rim-.001 and tip[:,2].max()>floor-.001)
    heads=[i for i,s in enumerate(shapes(rose_name)) if 'stem' not in s['name'].lower()]
    head_envelopes=[transformed_poly(rose_name,i,geometry_frame) for i in heads]
    head_low=min(vertices[:,2].min()-error for vertices,_,error in head_envelopes)
    head_clear=bool(head_low>rim and r['bench_force_n']<.01)
    support_force = r['pair_forces'].get('target_cup',0.) if p.get('dynamic_target_cup') else r['fixture_force_n']
    if p.get('target_filler_names'):
        from .pearl_pouring import cup_contains
        target_inverse = np.linalg.inv(pose_transform(rows['target_cup']['pose']))
        contained = {name for name in p['target_filler_names']
            if cup_contains((target_inverse @ pose_transform(rows[name]['pose']))[:3,3],
                            radius=p['filler_radius_m'],receptacle=cup_name)}
        connected = {'target_cup'}
        while True:
            added = {name for name in contained-connected if any(
                max(rows[name]['pair_forces'].get(other,0.),rows[other]['pair_forces'].get(name,0.))>.005
                for other in connected)}
            if not added:
                break
            connected.update(added)
        support_force += sum(r['pair_forces'].get(name,0.) for name in connected-{'target_cup'})
    supported=bool(support_force>p['min_support_n'] and
                   (not p.get('dynamic_target_cup')
                    or 'target_cup' in p.get('fixed_articles', ())
                    or rows['target_cup']['bench_force_n']>.005))
    active=inside and head_clear and supported and released(r)
    done=stable_outcome(e,'rose',[r],active)
    return done, {'stem_end_inside':inside,'depth_m':float(rim-tip[:,2].min()),
                  'flower_clear_of_table':head_clear,'head_above_rim':head_clear,
                  'fixture_supported':supported,'rigid_flower':True,'collider_clearance_m':float(margin),
                  'geometry':'four native bent stem capsules and native bud colliders',
                  'grasp_lift_recorded':'rose' in e.grasped,
                  'stability':'position and tilt; axial spin ignored if containment persists'}, ''


RULES={'L1-01':marker,'L1-02':plug,'L1-03':hat,'L1-04':cups,'L1-05':flower}
