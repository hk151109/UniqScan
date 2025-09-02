from gradio_client import Client

client = Client("SzegedAI/AI_Detector")

texts = [
        "The present book is intended as a text in basic mathematics. As such, it can have multiple use: for a one-year course in the high schools during the third or fourth year (if possible the third, so that calculus can be taken during the fourth year); for a complementary reference in earlier high school grades (elementary algebra and geometry are covered); for a one-semester course at the college level, to review or to get a firm foundation in the basic mathematics necessary to go ahead in calculus, linear algebra, or other topics.",
        "The present book is intended as a text in basic mathematics. As such, it can have multiple use: for a one-year course in the high schools during the third or fourth year (if possible the third, so that calculus can be taken during the fourth year); for a complementary reference in earlier high school grades (elementary algebra and geometry are covered); for a one-semester course at the college level, to review or to get a firm foundation in the basic mathematics necessary to go ahead in calculus, linear algebra, or other topics. Years ago, the colleges used to give courses in “ college algebra” and other subjects which should have been covered in high school. More recently, such courses have been thought unnecessary, but some experiences I have had show that they are just as necessary as ever. What is happening is that thecolleges are getting a wide variety of students from high schools, ranging from exceedingly well-prepared ones who have had a good first course in calculus, down to very poorly prepared ones.",
        "Nice — you’ve built a very thorough training script. Overall it’s solid and mostly correct, but I found a few bugs, potential pitfalls, and suggestions that will make the script more robust, reproducible, and correct (especially metric handling and a couple small lifecycle issues). Below I’ll list the problems (with why they matter) and give minimal, copy-pasteable fixes. I’ll mark the critical ones first.",
        "Artificial intelligence is rapidly evolving and has the potential to revolutionize various industries. From healthcare to finance, AI is being integrated to improve efficiency and outcomes."
]

for text in texts:
    result = client.predict(
            text=text,
            api_name="/classify_text"
    )
    print(result)