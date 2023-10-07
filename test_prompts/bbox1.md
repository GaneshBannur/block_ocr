# Prompt  
## System
Give the best possible answer and nothing else. Always explain the reasoning behind your answers in detail.
## User
Give the best ordering of the following strings as a single string. Use bounding boxes given only it isn't
possible to order using the meaning of the strings. Explain, in detail, the reasoning behind the order you give.
String: '''hello ''' Bounding Box: [[1 -7] [0 -16] [4 -7] [4 -16]]
String: '''how ''' Bounding Box: [[0 -6] [0 -10] [5 -6] [5 -11]]
String: '''are? ''' Bounding Box: [[0 0] [0 -5] [5 0] [5 -5]]  

# Response  
how hello are? 

The reasoning behind this order is based on the bounding boxes provided. The bounding box for "how" has the highest negative y-coordinate, indicating that it should be placed first. The bounding box for "hello" has the second highest negative y-coordinate, indicating that it should be placed second. Finally, the bounding box for "are?" has the lowest negative y-coordinate, indicating that it should be placed last.