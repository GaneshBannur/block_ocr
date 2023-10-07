def your_task(list of strings with bounding box of each string in the source image):
    if enough strings are meaningful to form a meaningful sentence using all strings and without using any bounding boxes:
        form the most meaningful sentence possible using all strings without using any bounding boxes
    else:
        give the best spatial ordering of the strings using the bounding boxes provided
        best spatial ordering of the strings can be based on x or y coordinates or both depending on which coordinate has more variation
    return a single string with spaces inserted between input strings
your_task([
    String: '''doing''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''are''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
])


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

def your_task(list of pairs of strings and their bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        best spatial ordering can be based on x coordinates or y coordinates or both coordinates depending on which coordinate has most variance
    return single string with spaces between input strings
your_task([
    String: '''doing''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''are''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
])


def your_task(list of strings with corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both based on which has most variance
    return single string with spaces between input strings

your_task(
    String: '''doing''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''are''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
)

def your_task(list of strings with corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both based on which has most variance
    return single string with spaces between input strings

your_task([
    {'String': '''doing''', 'Bounding Box': [[1, 9], [0, 0], [4, 9], [4, 0]]},
    {'String': '''you''', 'Bounding Box': [[0, 10], [0, 6], [5, 10], [5, 5]]},
    {'String': '''are''', 'Bounding Box': [[0, 16], [0, 11], [5, 16], [5, 11]]},
    {'String': '''how''', 'Bounding Box': [[1, 16], [0, 20], [4, 11], [4, 20]]}
])


def your_task(list of dicts containing strings and corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both based on which has most variance
    return single string with spaces between input strings

your_task([
    {'String': '''doing''', 'Bounding Box': [[1, 9], [0, 0], [4, 9], [4, 0]]},
    {'String': '''you''', 'Bounding Box': [[0, 10], [0, 6], [5, 10], [5, 5]]},
    {'String': '''are''', 'Bounding Box': [[0, 16], [0, 11], [5, 16], [5, 11]]},
    {'String': '''how''', 'Bounding Box': [[1, 16], [0, 20], [4, 11], [4, 20]]}
])


def your_task(list of dicts containing strings and corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both based on which has most variance
    return single string with spaces between input strings

your_task([
    {'String': '''bye''', 'Bounding Box': [[1, 9], [0, 0], [4, 9], [4, 0]]},
    {'String': '''oh im fine thank you''', 'Bounding Box': [[0, 10], [0, 6], [5, 10], [5, 5]]},
    {'String': '''thats good to hear''', 'Bounding Box': [[0, 16], [0, 11], [5, 16], [5, 11]]},
    {'String': '''how are you doing today''', 'Bounding Box': [[1, 16], [0, 20], [4, 11], [4, 20]]}
])

def your_task(list of dicts containing strings and corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both based on which has most variance
    return single string with spaces between input strings

your_task(
    String: '''bye''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''oh im fine thank you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''thats good to hear''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how are you doing today''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
)

def your_task(list of dicts containing strings and corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both depending on which has most variance
    return single string with spaces between input strings

your_task(
    String: '''bye''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''oh im fine thank you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''thats good to hear''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how are you doing today''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
)

def your_task(list of strings with corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x coordinates or y coordinates or both depending on which has most variance
    return single string with spaces between input strings

your_task(
    String: '''bye''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''oh im fine thank you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''thats good to hear''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how are you doing today''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
)

def your_task(list of strings with corresponding bounding boxes):
    if meaningful sentence can be formed using strings:
        form most meaningful sentence
    else:
        form best spatial ordering of strings using bounding boxes
        spatial ordering can be based on x or y coordinates or both depending on which has most variance
    return single string with spaces between input strings

your_task(
    String: '''bye''' Bounding Box: [[1, 9] [0, 0] [4, 9] [4, 0]]
    String: '''oh im fine thank you''' Bounding Box: [[0, 10] [0, 6] [5, 10] [5, 5]]
    String: '''thats good to hear''' Bounding Box: [[0, 16] [0, 11] [5, 16] [5, 11]]
    String: '''how are you doing today''' Bounding Box: [[1, 16] [0, 20] [4, 11] [4, 20]]
)
