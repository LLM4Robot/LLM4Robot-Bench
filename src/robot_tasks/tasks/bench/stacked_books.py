"""Streaming contact evidence for the real L3-03 stack-transfer/360-degree task."""
import math
import numpy as np
from ...robots.transforms import pose_transform
from .evaluation import released, sustained
from .push_triangle import containment_margin


def grasp_contact(row, arm):
    """Opposing measured finger forces distinguish squeezing from same-face pushing."""
    forces = np.asarray(row['finger_force_vectors'][arm], dtype=float)
    if forces.shape != (2, 3) or not np.isfinite(forces).all():
        raise ValueError('Invalid L3-03 finger force vectors')
    norms = np.linalg.norm(forces, axis=1)
    return bool(min(norms) > .05 and np.dot(forces[0], forces[1]) < -.5 * np.prod(norms))


def score(e, rows):
    p = e.spec.parameters
    names = p['stacked_book_names']
    lower_only = p.get('boundary_reference', 'both_books') == 'lower_book'
    lower, upper = (rows[n] for n in names)
    articles = {a.name: a for a in e.spec.design.articles}
    transforms = {n: pose_transform(rows[n]['pose']) for n in names}
    lt, ut = (transforms[n] for n in names)
    margins = {n: containment_margin(rows[n]['pose'], articles[n].goal,
               p['book_vertices'][n], p['target_vertices']) for n in names}
    lc, uc = (np.asarray(rows[n]['corners']) for n in names)
    supported = bool(lower['bench_force_n'] > .005)
    relative = (ut[:3, 3] - lt[:3, 3]) @ lt[:3, :3]
    bounds = np.asarray(articles[names[0]].asset_bounds)
    centre_over = bool(np.all(relative[:2] >= bounds[0, :2]) and
                       np.all(relative[:2] <= bounds[1, :2]))
    gap = float(uc[:, 2].min() - lc[:, 2].max())
    book_contact = max(lower['pair_forces'].get(names[1], 0),
                       upper['pair_forces'].get(names[0], 0)) > .005
    stacked = bool(book_contact and centre_over and -.003 <= gap <= .005
                   and lt[2, 2] > .95 and ut[2, 2] > .95)
    lifted = bool(lc[:, 2].min() > p['table_z'] + p['max_lift_m'] or
                  gap > p['max_lift_m'])
    # Norms alone mistake two fingers pushing the same face for a grasp.
    grasped = any(grasp_contact(rows[n], arm) for n in names for arm in e.spec.arms)
    contacts = {arm: any(max(rows[n]['fingers'].get(arm, [0, 0])) > .005
                         for n in names) for arm in e.spec.arms}
    yaw = {n: math.atan2(transforms[n][1, 0], transforms[n][0, 0]) for n in names}
    xy = np.asarray(lower['pose'][:2])
    state = getattr(e, '_book_transfer', None)
    if state is None:
        start_pose = (*p['start_centre_xy'], p['table_z'], 1., 0., 0., 0.)
        start_margins = {n: containment_margin(rows[n]['pose'], start_pose,
                         p['book_vertices'][n], p['start_vertices']) for n in names}
        state = dict(start_margins=start_margins, initial_xy=xy.copy(), xy=xy.copy(),
                     yaw=yaw, net={n: 0. for n in names}, common_net=0., positive=0., negative=0.,
                     contacts=contacts, arm_distance={a: 0. for a in e.spec.arms},
                     joint_distance=0., max_transfer=0., violations=[],
                     rotation_complete=False)
        e._book_transfer = state
        start_inside = (start_margins[names[0]] >= -1e-6 if lower_only
                        else all(v >= -1e-6 for v in start_margins.values()))
        if not start_inside:
            state['violations'].append(('The lower book' if lower_only else 'Both books')
                                       + ' must initially lie inside the start rectangle')
    violation_samples = [
        ('book_grasp', grasped, 'Grasping a book is forbidden, even without lifting'),
        ('book_lift', lifted, 'Books must not be lifted from the table or each other'),
        ('book_unstacked', not stacked, 'Books must remain stacked throughout transfer'),
        ('book_unsupported', not supported, 'The book stack must remain table-supported'),
    ]
    for key, active, reason in violation_samples:
        if sustained(e, key, active, seconds=p['constraint_debounce_s']):
            if reason not in state['violations']:
                state['violations'].append(reason)
    deltas = {n: math.atan2(math.sin(yaw[n] - state['yaw'][n]),
                          math.cos(yaw[n] - state['yaw'][n])) for n in names}
    if any(abs(d) > math.pi / 2 for d in deltas.values()):
        reason = 'Rotation sampling discontinuity; a full turn cannot be verified'
        if reason not in state['violations']:
            state['violations'].append(reason)
    valid = supported and stacked and not lifted and not grasped and not state['violations']
    distance = float(np.linalg.norm(xy - state['xy']))
    state['max_transfer'] = max(state['max_transfer'], float(np.linalg.norm(xy-state['initial_xy'])))
    if valid:
        # Signed unwrapping cancels back-and-forth rocking. Require BOTH books
        # to achieve 2*pi in the same direction at the same point in the episode.
        for n in names:
            state['net'][n] += deltas[n]
        # Only simultaneous, same-direction turning of both bodies advances the
        # stack. Rotating each book separately through 360 degrees must not pass.
        common_delta = (math.copysign(min(abs(v) for v in deltas.values()), deltas[names[0]])
                        if deltas[names[0]]*deltas[names[1]] > 0 else 0.)
        state['common_net'] += common_delta
        state['positive'] = max(state['positive'], state['common_net'])
        state['negative'] = max(state['negative'], -state['common_net'])
        # Translation or rotation with measured gripper contact counts as pushing.
        # Convert the common angular movement to arc length at a 5 cm radius.
        common_turn = min(abs(v) for v in deltas.values()) if deltas[names[0]]*deltas[names[1]] >= 0 else 0.
        movement = distance + .05 * common_turn
        for arm in e.spec.arms:
            if contacts[arm] and state['contacts'][arm]:
                state['arm_distance'][arm] += movement
        if all(contacts.values()) and all(state['contacts'].values()):
            state['joint_distance'] += movement
        if max(state['positive'], state['negative']) >= p['required_stack_turn_rad'] - 1e-6:
            state['rotation_complete'] = True
    state.update(xy=xy.copy(), yaw=yaw, contacts=contacts)
    both_pushed = all(v >= p['push_translation_m'] for v in state['arm_distance'].values())
    jointly_pushed = state['joint_distance'] >= p['joint_push_distance_m']
    transferred = state['max_transfer'] >= p['push_translation_m']
    if both_pushed and jointly_pushed:
        e.flags.add('both_grippers_pushed_stack')
    all_inside = (margins[names[0]] >= -1e-9 if lower_only
                  else all(v >= -1e-9 for v in margins.values()))
    # The real prompt does not require release or a particular final yaw.
    complete = bool(all_inside and valid and state['rotation_complete'] and
                    both_pushed and jointly_pushed and transferred)
    details = dict(books_in_target=sum(v >= -1e-9 for v in margins.values()),
        required_books=2, footprint_boundary_margin_m=margins,
        initial_footprint_margin_m=state['start_margins'],
        started_in_start_rectangle=(state['start_margins'][names[0]] >= -1e-6 if lower_only
                                   else all(v >= -1e-6 for v in state['start_margins'].values())),
        boundary_reference_book=names[0] if lower_only else 'both_books',
        lower_book_in_target=margins[names[0]] >= -1e-9,
        upper_book_containment_required=not lower_only,
        lower_book_table_supported=supported, upper_book_supported_by_lower=stacked,
        book_contact=bool(book_contact), book_vertical_gap_m=gap,
        both_arms_required=True, push_distance_by_arm_m=state['arm_distance'].copy(),
        simultaneous_push_distance_m=state['joint_distance'], both_arms_push_witness=both_pushed,
        simultaneous_push_witness=jointly_pushed, transfer_distance_m=state['max_transfer'],
        signed_rotation_deg={n: math.degrees(v) for n, v in state['net'].items()},
        common_rotation_progress_deg=math.degrees(max(state['positive'], state['negative'])),
        required_rotation_deg=360., full_turn_observed=state['rotation_complete'],
        grasped_now=grasped, lifted_now=lifted,
        released=all(released(rows[n]) for n in names), release_required=False,
        constraint_violations=list(state['violations']),
        orientation_requirement='No final yaw or centering requirement; full turn during transfer required')
    return complete, details, '; '.join(state['violations'])
