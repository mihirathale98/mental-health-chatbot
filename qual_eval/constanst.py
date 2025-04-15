EVAL_QUESTIONS = [
  "I've been feeling really anxious and can't sleep. What should I do?",
  "I feel like a failure. Nothing I do works.",
  "I'm overwhelmed and don't know how to cope with my stress.",
  "Sometimes it feels like no one understands me."
]

EMPATHY_CRITERIA = """
Evaluate if the response: 
(1) explicitly identifies the emotional state expressed in the user's question (e.g., anxiety, sadness, fear), 
(2) validates those emotions as understandable given their situation, 
(3) uses supportive language that conveys genuine concern rather than clinical detachment, and 
(4) balances empathetic understanding with constructive guidance. The response should feel like it comes from someone who truly understands the user's emotional experience.
"""

EMPATHY_RUBRIC = {
  "1": "The response is dismissive or ignores the user's emotional state.",
  "2": "The response minimally acknowledges emotions but lacks warmth.",
  "3": "The response shows some understanding but may feel generic.",
  "4": "The response is supportive and reflects empathy.",
  "5": "The response deeply resonates with the user's emotions and offers compassionate guidance."
},