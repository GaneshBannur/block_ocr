# Comments
Rationale is that explicit conditional execution is best understood by GPT when given as code since it has seen it there. Python syntax was used because it's the closest to english while still being a programming language. The space at the end of the user prompt is important.

# Prompts
## System
Give the best possible answer and nothing else.
## User
```python
def your_task(list of strings with bounding box of each string in the source image):
    if enough strings are meaningful to form a meaningful sentence using all strings and without using any bounding boxes:
        form the most meaningful sentence possible using all strings without using any bounding boxes
    else:
        give the best spatial ordering of the strings using the bounding boxes provided
        best spatial ordering of the strings can be based on x or y coordinates or both depending on which coordinate has more variation
    return a single string with spaces inserted between input strings
your_task([
    String: '''doin''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''are''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
])

```

# Response
how are you doin