# Comments
The prompt says use only meanings and no bounding boxes in the if condition. Space at end of user prompt is important.

# Prompt
## System
Give the best possible answer and nothing else.
## User
```python
def your_task(list of pairs of strings and their bounding boxes):
    if a meaningful sentence can be formed using all strings and without using any bounding boxes:
        form the most meaningful sentence possible using all strings without using any bounding boxes
    else:
        form the best spatial ordering of the strings using the bounding boxes provided
        best spatial ordering of the strings can be based on x coordinates or y coordinates or both coordinates depending on which coordinate has the most variance
    return a single string with spaces between input strings
your_task([
    String: '''doing''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''are''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
])

```

# Response
how are you doing