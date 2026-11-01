"""Explicit before/current RGB pairs, with no segmentation or object truth."""

from io import BytesIO

from PIL import Image, ImageDraw


def comparison_images(observation, max_edge=768):
    current = {frame.camera: frame for frame in observation.rgb}
    for reference in observation.grasp_references:
        for before in reference.rgb:
            after = current.get(before.camera)
            if after is None:
                continue
            images = [
                Image.frombytes("RGB", (f.width, f.height), f.data)
                for f in (before, after)
            ]
            for image in images:
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            # Original RGB is kept unobscured so grid labels do not cover a small object.
            width, height = max(i.width for i in images), max(i.height for i in images)
            canvas = Image.new("RGB", (width * 2, height + 28), "#202020")
            draw = ImageDraw.Draw(canvas)
            for index, (image, frame) in enumerate(zip(images, (before, after))):
                canvas.paste(image, (index * width, 28))
                title = "BEFORE CLOSE" if index == 0 else "CURRENT"
                draw.text(
                    (index * width + 4, 5),
                    f"{title} {frame.camera} t={frame.timestamp_s:.3f}s",
                    fill="white",
                )
            stream = BytesIO()
            canvas.save(stream, format="PNG")
            yield (
                {
                    "modality": "grasp_comparison",
                    "arm": reference.arm,
                    "camera": before.camera,
                    "reference_time_s": before.timestamp_s,
                    "current_time_s": after.timestamp_s,
                    "reference_camera_pose": before.calibration[
                        "T_world_camera_pose_wxyz"
                    ],
                    "current_camera_pose": after.calibration[
                        "T_world_camera_pose_wxyz"
                    ],
                    "meaning": "Left: before closing. Right: current. Camera may have moved; compare object relative to fingers and table. Depth-query coordinates refer ONLY to separate current RGB images, not this comparison.",
                },
                stream.getvalue(),
            )
