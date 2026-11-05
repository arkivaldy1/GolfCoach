from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced golf coach. Give practical, specific "
               "advice. Keep responses 3 to 4 sentences."),
    ("human", "I am struggling with {problem}. What should I focus on?"),
])

model = get_llm("ollama")
parser = StrOutputParser()

# The Chain
chain = prompt | model | parser

if __name__ == "__main__":

    """
    result = chain.invoke({"problem": "slicing my driver"})
    print(result)

    To address slicing, I recommend focusing on aligning your feet, knees, and shoulders parallel to each other, as this is a common cause of slicing. Additionally, ensure your grip is neutral, with your hands positioned equally on the club, and your wrists firm but not stiff. Practice swinging with a slight draw bias by rotating your hips and uncocking your wrists, which will help to counteract the slice.
    """



    """
    Batch Calls:
    problems = ["slicing my driver", "putting speed", "hitting out of bunkers"]
    results = chain.batch([{"problem": p} for p in problems])

    for p, r in zip(problems, results):
        print(f"=== {p} === \n{r}\n")

    === slicing my driver === 
    To address slicing, focus on ensuring a square face at impact. This can be achieved by checking your alignment and ensuring your feet, knees, and shoulders are parallel to the target line. Additionally, pay attention to your swing path, aiming for a more neutral or draw-biased path to reduce the likelihood of slicing. Practice swinging with a mirror or video analysis to help you develop a more consistent and square face.

    === putting speed === 
    To improve your putting speed, focus on aligning your putter face with the target line and making a smooth, pendulum-like stroke. Pay attention to the speed of your backswing and transition, aiming for a consistent tempo that allows you to make a controlled, rolling stroke. Practice using a metronome or a tempo trainer to help you develop a consistent rhythm, and also pay attention to your weight transfer and balance during the stroke.

    === hitting out of bunkers === 
    When hitting out of bunkers, focus on making a smooth, controlled swing, avoiding any twisting or turning of the hips and shoulders. Keep the clubface square to the target line and aim to hit the sand just above the ball, allowing the club to escape the bunker with a high trajectory. Practice hitting bunker shots with a slightly higher lofted club, such as a sand wedge or lob wedge, to help you escape the sand with more consistency.
    """

    for chunk in chain.stream({"problem": "mental game under pressure"}):
        print(chunk, end="", flush=True)
