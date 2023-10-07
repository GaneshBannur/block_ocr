# Comment
The two prompts given below show the difference between asking GPT to "give the best ordering" and "form the most meaningful sentence". The best ordering according to GPT need not be based on meaning. Also note that two of the bounding boxesw were given as the same so that it couldn't use those to order.

# Prompt
## System
Give the best possible answer and nothing else. Always explain the reasoning behind your answers in detail.
## User
Give the best ordering of the following strings as a single string. Use bounding boxes given only if it isn't possible to use the meaning of the strings. Explain, in detail, the reasoning behind the order you give.
String: '''hello ''' Bounding Box: [[1 9] [0 0] [4 9] [4 0]]
String: '''how ''' Bounding Box: [[1 9] [0 0] [4 9] [4 0]]
String: '''are? ''' Bounding Box: [[0 16] [0 11] [5 16] [5 11]]

# Response
The best ordering of the given strings as a single string is: "how are? hello"

The reasoning behind this order is as follows:

1. "how" comes before "hello" because it appears earlier in the text.
2. "are?" comes after "how" because it appears after "how" in the text.
3. "hello" comes last because it appears last in the text.

Since the bounding boxes are not necessary to determine the order in this case, we can ignore them. The order is solely based on the order of appearance in the given strings.

# Prompt
## System
Give the best possible answer and nothing else. Always explain the reasoning behind your answers in detail.
## User
Form the most meaningful sentence from the following strings. Use bounding boxes given only if it isn't possible to use the meaning of the strings. Explain, in detail, the reasoning behind the order you give.
String: '''hello ''' Bounding Box: [[1 9] [0 0] [4 9] [4 0]]
String: '''how ''' Bounding Box: [[1 9] [0 0] [4 9] [4 0]]
String: '''are? ''' Bounding Box: [[0 16] [0 11] [5 16] [5 11]]

# Response
Hello, how are? 

The reasoning behind this order is as follows:

1. "Hello" is placed first because it is a common greeting and is often used to initiate a conversation.
2. "How" is placed second because it is a question word and can be used to inquire about someone's well-being or state.
3. "Are?" is placed last because it completes the sentence by forming a question. The question mark indicates that it is a question.