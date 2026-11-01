> 历史归档：本文保留当时的设计和验证记录，旧命令、参数与任务编号不代表当前版本。当前入口见 [项目 README](../../README.md)。

# GPT-6 物理交互任务 Prompt（早期候选稿）

本文保留早期任务池的候选提示，不是当前执行清单，也不是运行时自动加载的 Prompt。任务名称、道具和分层可能已调整，例如 L1-01 已改为马克笔入笔筒，L2-06 已改为从承台夹起软橘子瓣 5 厘米，形变只记录，不再作为失败阈值。当前以 [共享操作说明](../../src/robot_tasks/policies/operating_notes.py)、[任务 TOML](../../configs/tasks/) 和每次运行保存的实际指令为准。不要把下面的旧候选提示直接覆盖到现有任务。

全部可运行任务由 `GPTPolicy` 自动加入公共闭环规则：理解目标与手臂角色，简要拆解阶段，再依据最新观测执行、核实和纠错；未确认抓持或对准就不推进，重置后重新确认，最终按当前任务的释放、握持和稳定要求申请停止。下面的共用约定仅用于说明，维护时以代码中的统一版本为准。

目标是在任务原有成功标准与资源约束内，提高完整完成任务的机会。以下是首版候选提示，尚未通过实测证明达到最高成功率。它们包含操作建议，结果应标注为“有操作提示的测评”，不与只有一句目标指令的结果混用。

范围依据本次对话明确给出的任务：L1 原有 8 项及新增柔软花茎插花，L2 保留 7 项，L3 原有 8 项及从 L2 移入的擦桌子，共 25 项。沿用已有编号；插花和擦桌子的新编号待确认。未取得最终名称与定义的额外 5 项 L3、未定义的 L0，不补造。此文件不代表飞书最新任务表已全部同步，也不预设各层实际成功率必然递减。

提示结构采用明确目标、必要上下文和可检验结果，并控制无关约束；依据 [OpenAI Docs：Prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) 与 [GPT-6 Astra 提示设计说明](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。机器人操作建议是结合本项目任务定义与当前接口设计的候选方案，不是这些资料已验证的机器人成功率结论。

## 使用方法

模型端统一使用英文：发送英文的“共用执行约定＋当前任务正文”，不要把整份任务库发送给模型。代码块内为实际 Prompt；中文标题、“运行前填写”和“实现条件”供实验人员查看，不发送给模型。

将正文中的 `{{parameter_name}}` 替换为本轮实际值；物体描述、目标要求和文字反馈也统一用英文，数值保留约定单位，物体 ID、相机名、字段名和接口枚举保留原值。缺省信息用明确英文填写，例如 `not provided; infer from the available observations`，不能保留占位符直接运行。语言统一是实验一致性约定，不表示已证明英文优于中文；若比较语言，应单独设计对照实验。

这些参数用于指明物体、目标、允许操作和公开的完成标准，不提供隐藏的物体真值坐标、最佳抓点或评分结果。描述目标可用名称、可见标记或实际提供的参考图。若参数没有确定，先完成任务定义，不让模型猜测成功阈值。

现有 `GPTPolicy` 已提供机器人身份、坐标／单位约定和 JSON 动作协议。本文件作为任务 `instruction` 的候选内容，不替换该协议。当前 GPT 下发稀疏 TCP 目标、参考欧拉角和归一化夹爪开口，并可查询已提供的 RGB-D 像素深度；关节轨迹由底层规划器生成。没有目标力／力矩、显式速度、触觉或六维力反馈。改变夹爪开口不等于指定夹持力，减小位移步长也不等于有速度控制。

力控类任务的提示分清“操作目标”和“接口实现”：仅凭 RGB 与位姿可尝试可见的稳定操作，不能据此报告精确控力能力；需要力信号的任务必须先接入真实可用的观测与控制能力。文中的实现条件均不是已完成声明。

当前已接入 16 个 L1—L3 仿真任务和 `run GPT-PICK-PLACE`；用 `python tools/launch.py list` 查看实际清单，唯一手动集成入口仍是 `check gpt-pick-place`。本文保留的其他候选任务不代表已有实现，不能只替换 instruction 就视为已实现。本文中的动作建议不能靠文字新增接口字段，候选稿与实际运行条件必须分开记录。

## 共用执行约定

```text
You control an ARX R5 robot with two-finger grippers to complete the task below. Use the provided RGB images, robot state, execution feedback, and available history to choose grasp locations, approach directions, action order, and recovery actions. Use only the arms and action interfaces enabled for this episode.

Return only the JSON required by the system action contract, without code, undeclared fields or text outside the JSON. Every decision requires a brief English note within 512 characters: current stage, observed evidence or mismatch, intended action and expected visible result. Briefly name planned stages in the first note without spending a separate planning-only call. Follow the declared coordinate frames, units, and action limits. TCP positions are absolute world targets; orientation uses the declared sparse Euler-angle reference, not a quaternion action field. Null pose components retain latest measured values; a null gripper retains its drive target. request_stop only asks for termination: an unsuccessful request is rejected, consuming a call, and does not automatically open grippers or retract the robot.

Before each decision, compare the last action's expected outcome with current RGB-D views, measured robot state and execution feedback. Advance only when prerequisites are observed; previous notes and completed commands are not object-progress evidence. Use depth queries and permitted observation-arm moves when geometry is unclear. Check that queried surfaces belong to the visible part rather than the background. Preserve support and follow arm-role restrictions. Do not invent force, tactile signals or hidden object coordinates. After a reset discard old grasp, alignment and completion assumptions and inspect the new scene.

Make useful progress in clearly free space. Near objects, openings, edges, or changing contacts, use smaller translation or rotation increments and inspect feedback before proceeding. IK does not guarantee straight-line motion or collision avoidance; maintain clearance for the grippers, objects, and the other arm. Preserve any support that is still needed until a replacement support has been established.

A completed joint target does not establish task success. During a grasp, contact may prevent the fingers from reaching the requested closing position, so executing does not by itself mean grasp failure. Assess object motion, gripper state, and actual feedback together. If expected progress is absent, make an evidence-based change to alignment, grasp, or approach instead of repeatedly advancing the same blocked target.

Actively complete subsequent steps and recoverable errors within the episode budget. After repeated rejected targets, reobserve and revise the pose or clearance instead of extending speculative detours; planner failure does not establish a particular obstacle. Before requesting stop, inspect the task's required final support, release or continued holding, and stability. Not every task requires release. Use simulation timestamps to verify durations. Empty commands with request_stop=false hold existing drive targets; first ensure those targets will not continue squeezing or pushing. Request stop with empty commands only after checking completion; the runner enforces total budgets. An unsuccessful stop is rejected and the independent evaluator determines success.
```

## L1：任务相关 Affordance 与几何操作

本组使用统一基础控制设置，主要保留自主选取操作部位、姿态与几何对准。提示提供操作原则，不指定抓点真值或预制轨迹。插花为新增柔性候选，最终分层需结合实际物体确认。

### L1-02 异形块嵌槽

运行前填写：目标块、目标槽、就位高度／表面关系、保持时间。实现条件：轻放配合，槽的碰撞几何保留凹形结构。

```text
Fully seat {{shaped_block_id}} in {{target_recess_id}}, satisfying {{seated_height_or_surface_relationship}}. After release and gripper withdrawal, it must remain stable for {{hold_time_s}} seconds without raised edges. This is a loose-fit placement task; do not achieve seating through press-fitting, deformation, or movement of the recess fixture.

Identify the matching outlines, asymmetric features, and front/back orientation. Choose a grasp that allows full seating. Align position and orientation using multiple views to reduce errors caused by perspective, then lower incrementally.

If one side contacts first, the block tilts, or lowering makes no progress, lift it clear of the constraint and correct its pose before trying again. Do not force it flat with the gripper. After release, inspect the edges relative to the recess surface and confirm full seating with stable support.
```

### L1-03 插头插接

运行前填写：插头、对应接口、插接到位标志、保持时间。实现条件：未通电、低插入力测试件；本任务不包含卡扣解锁或电气功能验证。

```text
Insert {{plug_id}} into {{matching_socket_id}} until {{seating_indicator}} is satisfied. After releasing and withdrawing the gripper, the connection must remain seated for {{hold_time_s}} seconds. The test connector is unpowered; keep the socket fixture in place.

Identify the plug's mating end, socket outline, keying orientation, and graspable housing. Avoid grasping pins, contacts, or locations that obstruct insertion travel. Align the mating axis and orientation before advancing incrementally. If a cable is present, leave enough slack to prevent it from pulling the plug out of alignment.

If only one side enters, the fixture moves, or advancement repeatedly fails, retract and check orientation and lateral alignment. Do not force an obviously incorrect orientation. Verify both the seating indicator and retention after release; touching the socket face is not completion.
```

### L1-04 薄片入槽

运行前填写：薄片、狭槽、接收区域标识。实现条件：刚性薄片与真实槽口碰撞，接收区可判定；不包含弯折薄片。

```text
Pass {{thin_plate_id}} completely through {{slot_id}} into {{receiving_region_id}}. Do not bend the plate, widen the slot, or bypass the slot through another opening.

Identify the plate's thickness direction and the slot's long axis. Choose a stable grasp that leaves room for the leading end to enter. Align the plate plane and insertion direction with the slot before approaching the entrance for smaller corrections.

Choose the release point according to whether the gripper can continue through the available space; do not wedge the gripper in the entrance. If the plate catches an edge or jams at an angle, retract while it remains controllable and realign rather than forcing it forward. Confirm that the entire plate has passed through the slot into the receiving region; partial insertion is insufficient.
```

### L1-05 钥匙插锁

运行前填写：钥匙、锁孔、插入深度／就位标记。实现条件：顺畅配合，锁体固定，仅插入。

```text
Insert {{key_id}} into {{keyhole_id}} to {{insertion_depth_or_seating_marker}}. Insert only: do not turn the key or attempt to unlock the lock. Keep the lock fixture fixed.

Identify the key handle, tooth or groove orientation, and keyhole outline. Choose a grasp that leaves the required insertion length exposed. Determine the correct face and rotational orientation, align the key axis with the keyhole, and advance incrementally.

If the tip does not enter, the orientation is incompatible, or insertion is blocked, retract outside the entrance and reassess orientation. Avoid levering the key sideways against the keyhole. Once the required position is reached, stop advancing and verify the final inserted state. Release only if separately required by the task.
```

### L1-06 工具柄定点抓取

运行前填写：工具、允许抓持的柄部判定、规定抓取方向、放置区域。实现条件：轻质刚性工具与柄部接触判定。

```text
Pick up {{tool_id}} and place it in {{placement_region_id}}. Gripper contact must lie within {{allowed_handle_region_definition}} and satisfy {{required_grasp_direction}}. The grasp must support the tool's subsequent normal use. Do not complete the transfer by grasping bristles, a scraper's working edge, or another functional end.

Distinguish the handle from the working end. Choose the specific grasp using the center of mass, available gripper clearance, and the intended placement. After closing, perform a short lift and check whether the tool moves stably with the gripper. If it slips or its orientation is unsuitable, restore stable support before adjusting the grasp.

Keep the working end clear of surrounding objects during transport. Establish stable support in the target region before releasing and withdrawing. Confirm that the tool has not rolled away and that the required handle region and grasp direction were used.
```

### L1-07 挂环上钩

运行前填写：刚性环、指定挂钩／钩口、保持时间。实现条件：开口刚性钩，无弹性卡扣，孔洞可真实穿过。

```text
Pass {{rigid_ring_id}} through {{specified_hook_and_opening_id}} and hang it. After gripper release, the hook must independently support the ring for {{hold_time_s}} seconds. Holding the ring up with the gripper is not a completed state.

Identify the ring opening, hook entrance, and load-bearing direction. Choose a grasp that leaves the threading path clear. Align the ring plane with the opening, then thread and seat it incrementally while checking clearance between the gripper and hook.

If the hook tip blocks the ring, withdraw along a reversible path and adjust orientation instead of using large swinging motions. Move the ring onto a load-bearing part of the hook, gradually transfer support, release, and withdraw. Use a new observation to confirm that the ring has not slipped out through the opening.
```

### L1-08 刚性积木定姿堆叠

运行前填写：积木标识、目标结构文字／实际提供的参考图、位置与朝向容差、保持时间。实现条件：支撑与稳定性评分；不能只用积木中心位置判断成功。

```text
Use {{block_set_ids}} to build {{target_structure_description_or_provided_reference_image}}, satisfying {{public_position_and_orientation_tolerances}}. After all grippers withdraw, the structure must stand independently for {{hold_time_s}} seconds.

Choose the assembly order, supporting faces, and grasps from the target structure. Build a stable base first. Before each placement, identify the block's bottom face, the supporting area below it, and the required orientation. Do not hide unresolved problems in lower layers by adding upper layers.

Lower incrementally near the supporting surface. Confirm support before releasing rather than dropping the block from above. After release, inspect position, tilt, and movement of lower blocks. If correction is needed, retrieve and replace the unstable block instead of continuing to build on a structure still held up by a gripper.
```

### 插花：柔软花茎、较重花头（新增，编号待确认）

运行前填写：实际花材、目标花瓶／瓶口、目标插入与花头位置、保持时间；由接口声明可用单臂或双臂。实现条件：先确认真实物体与柔性仿真，不能以刚性花茎替代。若持续形变修正成为主要难点，需重新审核 L1 归类。

```text
Place {{flower_id}} into {{vase_and_opening_id}}, satisfying {{required_insertion_and_flower_head_position}}. After robot release, the vase must naturally support the flower for {{hold_time_s}} seconds. The stem bends and droops like a rope, and the flower head is relatively heavy. Keep both intact.

Observe the flower head, receptacle, stem, and existing support relationships. Choose grasp or support locations that give control over the part being manipulated. Do not assume that holding an end far from the flower head will make the whole flower behave like a rigid rod. Avoid crushing petals or kinking the stem.

Plan around the available arms and support from the vase. If two arms are enabled, one may support a robust region near the flower head while the other guides the insertion end. With one arm, use permitted existing supports in stages; do not assume a second hand is supporting the flower. Account for the observed drooping shape, maintain control over a short segment near the insertion end, and feed it through the opening incrementally while transferring support.

If the stem droops out of control, kinks, or catches at the opening, restore support before changing the grasp or direction. Do not pull on the flower head to force insertion. Confirm that the stem is inside the vase, the head meets the target position requirement, and the vase is stable before gradually releasing and checking the final state.
```

## L2：稳定交互中的受力与运动调节

沿用老师修订后的边界：保留自主选择抓持与几何对准，利用物体自然存在的阻力、负载和易损性；建立主要操作关系后保持稳定。以下提示不预先给出最佳接触点。若正式协议明确提供抓点或方向，应把它列为模型可见的任务参数，而不是在后台替模型完成抓持。

### L2-01 双臂拔笔帽

运行前填写：马克笔、笔身与笔帽各自的放置区域。实现条件：双臂真实抓持和拔出阻力；当前开口控制不能直接指定夹持力。

```text
Remove the cap from {{marker_id}}. Place the marker body in {{marker_body_placement_region}} and the cap in {{cap_placement_region}}. Keep both parts intact without slipping or dropping them. Do not use the writing tip as a grasp or support point.

Identify the cap/body boundary and connection axis. Choose stable grasp locations for both arms that do not interfere with each other. Establish both grasps before incrementally increasing separation along the connection axis. Avoid gripper collisions and bending the marker.

Use relative motion in the images, gripper opening, and provided feedback to distinguish cap separation from the entire marker slipping within a grasp. If slipping occurs, restore a stable grasp first. If the axes are misaligned, correct alignment before continuing rather than advancing the same blocked target. Reduce subsequent motion once separation occurs, then place, release, and verify both parts.
```

### L2-02 双臂夹持力交接

运行前填写：被交接物、移交侧、接收侧、目标交接区域、独立持物时间。实现条件：双臂夹持与负载转移；定量评估夹持力还需实际力反馈／控制及评分。

```text
Transfer {{object_id}} from {{donor_arm}} to {{receiver_arm}} within {{handover_region}}. After the donor fully releases and withdraws, the receiver must hold the object independently and stably for {{independent_hold_time_s}} seconds. Do not drop, visibly deform, or damage the object.

Choose grasp locations and a handover pose that leave room for the receiving gripper. While the object remains reliably supported, approach with the receiver, close its gripper, and inspect the grasp relationship. Sending a closing command alone does not establish a successful receiving grasp.

Once the receiver has made contact and can constrain the object, gradually reduce the donor's grip and observe for slip within a recoverable range. Fully release the donor only after the receiver has established stable support. If the object slides or rotates, restore donor support before correcting the receiving grasp. Avoid pulling the object in opposing directions during load transfer.

Verify that the receiver holds the object independently and the donor is no longer providing hidden support. Use timestamps to confirm the required stable hold before ending.
```

### L2-03 固定上下层的单臂侧向抽书

当前实现与成功条件以 [任务配置](../../configs/tasks/l2_03_middle_book_extraction.toml) 为准。

道具：三份原生《神雕侠侣》闭合书本资产，220×140×20 mm，单本质量 378.9 g；一个固定蓝色支撑盒，140×220×70 mm。支撑盒为视频布置的尺寸估计，不是实物扫描。
布置：蓝色支撑盒与书堆放在双臂中间，中心为实验室坐标 (0.19565, 1.30) m；书本平放在蓝色盒上，上下书本固定，中层向第三视角画面右侧（世界 +X）突出 40 mm；不是向机械臂正面（+Y）突出。书堆居中，朝右突出的一侧由右臂接近和抽取。中层落稳后与固定上层留 5 mm 净空，上层重量不压在中层上。
抓法：右臂从突出侧接近，一指在上、一指在下，夹住 20 mm 厚度，避开上下固定书；不能夹 140 mm 宽度。左臂按需观察，不扶书。
完成：夹持后沿 +X 侧向抽取约 110 mm，直到整个后缘离开固定书堆；最终由手或桌面承托均可，不要求持续静止或额外等待，不需为验证重新抓起。悬空掉落中或掉桌不算完成。保留原生尺寸、质量和材质；不模拟翻页或书本弯曲。本次只对齐初态布置，未将目标改成真机视频中的仅抽出 5 cm。

```text
Use the RIGHT R5 arm to extract ONLY the middle book. The LEFT arm may
move its wrist camera for a clearer view, but must not touch the books.
The stack and its support box are centred between both arms.
The three books are stacked horizontally. Upper and lower books are fixed;
there is no need to support them. On a blue support box, the middle book
protrudes about 40 mm from the RIGHT side in the overview image, along world
+X. This is a sideways extraction, not a pull toward the robots along +Y.
Do not rotate or stand up the stack.

Inspect the exposed right-side edge and the space above and below it. Each book
is 220 mm long, 140 mm wide and 20 mm thick. The R5 opens at most 70 mm:
it CANNOT grasp across the 140 mm width. Orient the gripper so its pads close
across the 20 mm thickness, one above and one below the projecting middle
book. Place both pads entirely ahead of the upper/lower books, with enough
pad overlap to carry the 378.9 g book. Avoid the edges and the fixed neighbours.

Approach from the +X side toward the exposed edge with an open gripper; inspect complementary views
before closing in a separate decision. Verify actual pad contact and book
motion using a small outward pull, not a vertical lift against the upper book.
Pull sideways along world +X, maintaining the book height and orientation
while it remains between the fixed neighbours. Reobserve if blocked or slipping;
do not continue an empty grasp or increase off-axis motion.

Continue until the entire rear edge has cleared the fixed stack, approximately
110 mm of travel from the initial pose. Seeing just the front end move is not
completion. Once the book is fully extracted by grasping and pulling, it may
remain in the gripper OR rest on the tabletop. Either final state is sufficient;
no continued grasp, timed hold, release or target placement is required.
Do not regrasp a fully extracted book already supported by the table merely
to prove retention. Request stop after verifying complete extraction and
support by the hand or table; a book still falling unsupported is not complete.
Use available feedback; do not invent object poses or force measurements.
The fixed upper book leaves about 5 mm clearance above the settled middle
book; its weight is not applied to the middle book. This tests guided extraction,
not supporting or balancing the upper stack. There is no automatic scene reset.
```

### L2-04 纸杯间倒珍珠

本任务已替换原满载碗搬运。当前英文 prompt 与完整布置见[任务配置](../../configs/tasks/l2_04_cup_pearl_pouring.toml)。

使用两只现有纸杯，源杯装 200 颗 8 mm 珍珠，初始料面接近半杯高，装料杯约 75.8 g，接收杯固定在桌面。第三视角画面右侧为装料源杯，左侧为固定接收杯，中心相距 200 mm。右臂侧向抓取源杯、试提，在接收杯外先抬高，再边倾斜边平移以保持杯口对准和腕部净空；左臂仅在需要时观察，并避开倾倒区。提示仍要求尽量转移 160 颗；独立评分只要求至少 10 颗进入接收杯并停留，源杯放下松手、竖立受桌面支撑约 0.5 秒即完成；允许残留，洒落单独记录。固定接收杯不要求支撑力读数。源杯不绑定夹爪，纸杯按刚体模拟。


### L2-06 从圆柱承台夹起橘子瓣 5 厘米

当前完整英文提示词以 `configs/tasks/l2_06_orange_transfer.toml` 为准，公共规划、观测及反馈信息照常附加。

橘子放在双臂中间固定的 13 cm 高圆柱承台上，宽面平放，绕竖直轴转 90°，长轴沿世界 Y、第三视角画面纵向。左臂夹住较饱满的中部，允许弹性挤压和继续收紧；右臂只观察。将橘子相对初始静置位置提起至少 5 cm，拿着即可结束；建议提 7 cm，留出夹持沉降余量。没有绿色目标区、搬运、放下、额外等待或形变阈值。抬空爪、弹飞或仅在承台上推移不算抓起。

## L3：变化接触下的持续联合调整

本组保留自主抓持、几何对准和稳定交互，进一步根据啮合、形变、支撑与阻力变化持续调整。任务正文描述操作目标与可观察的恢复线索；尚未提供的力／力矩信息不能由模型自行假定。

### L3-03 双臂扶东方树叶瓶旋下松盖

现行仿真采用双臂中间的自由瓶身、预松盖；左臂扶持瓶身中下部，右臂旋下盖子。完整执行提示以 [任务 TOML](../../configs/tasks/l3_03_bottle_cap_unscrew.toml) 为准，公共规划与核验协议仍统一注入。

完成条件为左臂扶瓶时右臂取下瓶盖，瓶盖完全脱离、放到桌面并释放；瓶身可保持左臂扶持或释放后直立在桌面。不要求规定圈数或静止等待。不测紧盖开启力矩和防盗环破坏。

### L3-05 双臂旋下预松螺母

当前版本使用原生可拆螺母膨胀螺栓，横放在双臂中间，第三视角看主体朝左、螺母朝右，替代旧的粗螺丝旋入。左臂从桌面拿起并持稳套管主体，右臂夹住六角螺母，逆时针旋出；初始预松至剩约两圈。完整英文提示以 [任务 TOML](../../configs/tasks/l3_05_coarse_screw.toml) 为准，公共规划与核验协议统一注入。

实际旋出分离且双手分别持握即成功，不再要求入孔深度、固定旋转圈数、释放、指定摆放或额外等待。仅本任务采用 1920 Hz 与 10 mm 最小夹爪开口；不测紧固预载、首次啮合、滑牙或损伤。

### L3-06 剥橘子

运行前填写：橘子、允许使用的工具、去皮范围、果肉损伤限制、果皮放置区域。实现条件：果皮分离、断裂与果肉损伤可评估；R5 夹爪能否建立起剥边缘需先验证，不默认人工预剥或预切。

```text
Use {{allowed_tools_or_gripper_only}} to peel {{orange_id}} over {{required_peeling_extent}}, satisfying {{pulp_damage_limit}}. Place removed peel in {{peel_placement_region}}. Do not drop the fruit or use crushing or breaking apart the pulp as a way to remove the peel.

Observe the peel, natural openings or locations where a starting edge could be formed, and the fruit's current support. Choose how to stabilize the fruit and establish a starting point using only permitted tools. Do not assume an unseen pre-cut opening exists.

Once a peel edge can be grasped, work in small sections. Choose new grasp locations and directions according to the advancing separation boundary, keeping the action on the peel rather than the pulp. Adjust actions using visible deformation, separation progress, and actual feedback. At a resistant region, change the local direction or regrasp instead of pulling off a large section together with the pulp.

As more pulp is exposed, adjust support to avoid crushing it with the grippers. Check remaining peel and damage progressively. Stop once the required extent is reached, keep the fruit stable, and place the removed peel in its designated region.
```

### L3-07 贴纸／胶带剥离

运行前填写：贴纸／胶带、基底、剥离范围、完整性与基底损伤限制、允许工具、剥离物放置区域。实现条件：粘附与剥离前缘可变化；不起边版本不能默认为已起边。

```text
Using {{allowed_tools_or_gripper_only}}, peel {{sticker_or_tape_id}} from {{substrate_id}} over {{required_peeling_extent}}. Satisfy {{peeled_material_integrity_and_substrate_damage_limits}} and place the removed material in {{placement_region}}. Do not complete the task by moving the substrate out of view, cutting away the substrate, or tearing regions that must remain intact.

Check whether a graspable edge exists. If not, establish one using permitted actions rather than assuming it has already been lifted manually. Stabilize the substrate and avoid grasping it together with the edge.

After establishing a grasp, adjust pulling direction and peel angle incrementally according to the moving separation front. If only the substrate moves, the material stretches, or a local tear develops, restore support and adjust the grasp, direction, or increment rather than increasing the pull in the same direction.

When regrasping, use the separated portion to maintain control and avoid pressing it back onto the adhesive surface. Verify that the required region is actually peeled and that the substrate and removed material satisfy the integrity requirements before placing the removed material.
```

### L3-08 柔性布面拉链开合

运行前填写：拉链、打开或关闭的目标位置、端部是否已啮合／需自行接合、最终放置要求。实现条件：拉链滑块、链齿与柔性布面联动，不能把布料两边的位置接近直接判为闭合。

```text
Operate {{fabric_zipper_id}} to {{target_open_or_closed_position}}. The initial end-engagement condition is {{zipper_end_engagement_condition}}, and the required final placement is {{final_placement_requirement}}. Preserve the fabric, pull tab, and teeth. Avoid tooth separation and jamming; moving the entire fabric does not count as opening or closing the zipper.

Identify the pull tab, slider, local zipper direction, and usable fabric support regions. Choose how one arm grasps the tab while the other arm or a permitted existing support stabilizes the fabric. If the task requires joining the zipper ends, first align and engage them before pulling.

As the slider moves and the fabric deforms, adjust the pulling direction in segments to follow the current local zipper direction rather than continuing along the initial direction. Observe actual slider progress, fabric entering the slider, and excessive tension on either side. If a problem appears, first relieve the pull, retract slightly if appropriate, and restore support and direction before continuing.

Reduce increments near the endpoint. Verify tooth engagement or separation and the achieved opening length before releasing according to the required final state. A tab near the endpoint is insufficient if the teeth remain incorrectly separated or fabric is trapped.
```

### L3-09 带卡扣连接器解锁拔出

运行前填写：连接器、允许的解锁操作、连接器放置区域。实现条件：卡扣按压、解锁和拔出阻力联动；未通电测试件，基座可稳定支持。

```text
Use {{permitted_unlatching_operation}} to release the latch of {{connector_id}}, fully unplug it, and place it in {{placement_region}}. Keep the base in place and preserve the latch, mating interfaces, and cable. Breaking the latch to separate the connector is not permitted.

Identify the latch, graspable housing, withdrawal direction, and base support region. Choose how to stabilize the base, actuate the latch, and hold the connector. Avoid pulling on the cable to transfer load directly to the interface.

Use only controlled small test motions along the withdrawal direction. Assess release using visible latch displacement and actual relative motion. If the connector does not withdraw, first check whether pressure is applied at the correct location, release is maintained, and the axes are aligned. Do not simply increase the pulling target.

Maintain the contact needed for unlatching until the connector has passed the locking region, then adjust the grasp and withdrawal direction as the contact state changes. After separation, place it stably and confirm an intact latch, an unmoved base, and genuinely separated interfaces.
```

### L3-10 柔性密封圈入槽

运行前填写：密封圈、环槽、完整就位标准、允许拉伸与损伤限制。实现条件：柔性环形变、自接触、槽边接触与保持；夹爪／允许工具能进入局部装配区域。

```text
Install {{seal_ring_id}} completely into {{annular_groove_id}}, satisfying {{complete_seating_criterion}} and {{stretch_and_damage_limits}}. Avoid twisting, pinching, local unseating, and damage. Do not cut or glue the ring, or keep it pressed down with a gripper to create an apparent seated state.

Identify the ring's cross-sectional orientation, groove entrance, and currently seated and unseated sections. Choose the first locating point and assembly order. After establishing reliable local seating, guide adjacent unseated sections into the groove while preserving the seated portion's support.

Adjust local grasp locations, stretching direction, and pressing points according to deformation and groove-edge contact. Do not pull already seated sections out to install a distant section. If a section flips, overlaps, or becomes pinched, reduce tension, back out that local section, straighten it, and then resume installation.

Inspect the full circumference for unseated sections, changing the viewing direction when needed. Do not assume an unseen region is seated. Gradually release external support and check that the ring remains in the groove, with complete seating and preserved integrity.
```

### L2-07 板擦擦除绿板字迹（历史编号 L3-11）

现行版本见[任务配置](../../configs/tasks/l2_07_board_wiping.toml)：固定绿板在桌面内转 180°，书写面朝上，黑色 GPT6 字迹从公共第三视角看上下颠倒。板擦置于右侧被动支座上，两手初始空置；右臂抓取和擦拭，左臂观察。真实板擦接触和横向移动擦除至少 90% 初始字迹即可，不要求放回、松手或额外等待；属于接触清洁代理。

### L3-14 夹爪推三角块入目标区

现行版本见[任务配置](../../configs/tasks/l3_14_gripper_push_triangle.toml)：按参考照片关系，第三视角中块在目标右上方、更靠近机械臂，目标在左下方；两个三角尖端均朝机械臂。右夹爪推动、左臂观察。实际推入后整个块在宽容目标区内并受桌面承托即完成，不要求朝向一致或额外等待。

### L3-15 两块扫入固定簸箕

现行版本见[任务配置](../../configs/tasks/l3_15_bimanual_sweep_blocks.toml)：第三视角左簸箕、右手刷，两块在中间；簸箕开口朝右并固定。右臂持刷向左扫入，左臂观察。两个方块均在篮内、受到簸箕承托且有实际刷扫过程即可，不要求最后放下刷子或等待。

## 提示版本与验证

语言版本：英文模型输入，中文阅读说明。共用约定、25 项任务正文与模板变量名均已统一为英文；翻译保留原任务目标、成功条件和动作协议。

本文只提供首版候选提示，不记录虚构的成功率。先用少量独立预实验检查目标是否清楚、接口是否够用、模型是否把“动作完成”误当成“任务完成”；对有依据的失败修改提示，然后固定正式测试版本。保留失败、超时与未完成停止，不能只统计模型自报成功的回合。

比较两个 prompt 时，保持任务物体、初始条件分布、可见信息、动作接口和预算一致；以外部评分器判定的完整成功率为主，同时检查损伤、洒落、稳定性等任务约束。本文没有示范轨迹，属于带操作提示的指令方案，不等于已经开展 ICL 示范学习。



任务设计按照框架

任务设计原则、任务目标、成功条件、操作协议和可观察线索进行描述。每个任务正文包含：（自动化结果）

仿真跑出预期的结果（L1-L3 得分decay）

真机仿真对齐（任务层面）

主动交互（探索）






真机任务排期
机器到位
双臂gpt6接入
任务道具排布和测试
任务迭代和反馈


项目定位任务人员分配
