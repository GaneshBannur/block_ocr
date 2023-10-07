[DONE]1. Symlink all the configs in the `configs` directory so that when the original is updated somewhere else, so is the config in the `configs` directory.
[DONE]2. If a paragraph contains only one line then don't pass it to the reading order detection module.
[DONE]3. Fix the positioning of recognized text on the final image so that the text for each paragraph comes next to it. If paragraphs are not recognized properly then this may be messed up no matter what.
[DONE]4. Draw bounding boxes around paras recognized rather than lines.
[PROBABLY NOT NEEDED]5. Move the visualization of paragraphs in image to `end_to_end_ocr.py` rather than it being within the unified detector code.
5. Test on the images I took
