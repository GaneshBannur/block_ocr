# Prompt
## System
Give the best possible answer and nothing else. Always explain the reasoning behind your answers in detail.
## User
Always explain the reasoning behind your answers in detail.
Expected Input: List of strings with bounding box of each string in the source image
Your Task:
If enough strings are meaningful to form a meaningful sentence using all strings and without using any bounding boxes.
Then form the most meaningful sentence possible using all strings without using any bounding boxes.
Else give the best spatial ordering of the strings using the bounding boxes provided for the strings.
Expected Output: A single string.
Input:
String: '''great ''' Bounding Box: [[1 9] [0 0] [4 9] [4 0]]
String: '''is ''' Bounding Box: [[0 10] [0 6] [5 10] [5 5]]
String: '''ths ''' Bounding Box: [[0 16] [0 11] [5 16] [5 11]]

# Response
The given strings "great", "is", and "ths" do not form a meaningful sentence when combined. Therefore, we need to determine the best spatial ordering of the strings using the provided bounding boxes.

To determine the spatial ordering, we can consider the y-coordinate of the top-left corner of each bounding box. The string with the lowest y-coordinate will be placed first, followed by the string with the next lowest y-coordinate, and so on.

Looking at the given bounding boxes:
- Bounding box for "great" has a top-left corner at (1, 9)
- Bounding box for "is" has a top-left corner at (0, 10)
- Bounding box for "ths" has a top-left corner at (0, 16)

Based on the y-coordinates, the correct spatial ordering of the strings would be:
1. "ths"
2. "is"
3. "great"

Therefore, the best spatial ordering of the strings using the provided bounding boxes is "ths is great".