# Prompt
## System
    Give the best possible answer and nothing else.
## User
```python
def your_task(list of strings with corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering is based on x coordinates or y coordinates or both depending on which has most variance
    return single string with spaces between input strings

your_task(
    String: '''bye''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''oh im fine thank you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''thats good to hear''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how are you doing today''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
)

```