import re
import random

def simple_chatbot():
    print("ChatBot: Hello! I'm a simple rule-based chatbot. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ").lower()
        
        # Exit condition
        if user_input == 'byee':
            print("ChatBot: Goodbye! Have a nice day.")
            break
            
        # Greeting patterns
        elif re.search(r'hi|hello|hey|greetings', user_input):
            responses = ["Hello there!", "Hi! How can I help?", "Greetings!"]
            print(f"ChatBot: {random.choice(responses)}")
            
        # How are you patterns
        elif re.search(r'how are you|how\'s it going', user_input):
            print("ChatBot: I'm just a program, so I don't have feelings, but thanks for asking!")
            
        # Name patterns
        elif re.search(r'your name|who are you', user_input):
            print("ChatBot: I'm SimpleBot, a rule-based chatbot.")
            
        # Help patterns
        elif re.search(r'help|what can you do', user_input):
            print("ChatBot: I can respond to greetings, tell you about myself, and answer simple questions.")
            
        # Time patterns (very basic)
        elif re.search(r'time|what time is it', user_input):
            print("ChatBot: I don't actually have access to the current time. I am Sorry!")
            
        # Thank you patterns
        elif re.search(r'thank|thanks|appreciate', user_input):
            print("ChatBot: You're welcome!")
            
        # Default response if no patterns match
        else:
            print("ChatBot: I'm not sure I understand. Try asking me something simpler.")

# Start the chatbot
simple_chatbot()