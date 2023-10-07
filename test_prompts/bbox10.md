# Comment
The conditions to use all string and none of the bounding boxes have not been explicitly mentioned.

# Prompts
## System
Give the best possible answer and nothing else.
## User
```python
def your_task(list of pairs of strings and their bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence using strings
    else:
        form best spatial ordering of strings using bounding boxes
        best spatial ordering of strings can be based on x coordinates or y coordinates or both coordinates depending on which coordinate has most variance
    return single string with spaces between input strings
your_task([
    String: '''doing''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''are''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
])

```

# Response
how are you doing