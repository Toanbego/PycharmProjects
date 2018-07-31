
"""Add a question and answer in the string. seperate the two with a : """
questions = """Best album of all time? :Back in black
                Earths gravitation? :9.81 
                Who plays Aragorn in lord of the rings? :Viggo Mortensen 
                What is love? :Baby dont hurt me"""

listOfQuestions = []
listOfAnswers = []

#Split string into question and answer
split_string = questions.split('\n')
print(split_string)
for i in split_string:
    question, answer = i.split(':')
    listOfQuestions.append(question)
    listOfAnswers.append(answer)


def go_through_quiz(listOfAnswers, listOfQuestions):
    score = 0
    for i in range(len(listOfAnswers)):
        answer = input(listOfQuestions[i])

        if answer == listOfAnswers[i]:
            print('Du svarte riktig!')
            score += 1
        else:
            print('It is wrooooong')
    print('Du klarte',score, 'av totalt 4 mulige')
go_through_quiz(listOfAnswers, listOfQuestions)