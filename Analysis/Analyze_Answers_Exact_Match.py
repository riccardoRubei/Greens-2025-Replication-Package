import os
import json
import nltk
import pandas as pd

global correctnessVar
#base_directory = "../Test"

def loadAnswers(basePath):
    fileAnswers = {}
    for file in os.listdir(basePath):
        answers = []
        data = open(basePath + '/' + file, 'r')
        answer = []
        for line in data.readlines():

            if(">>>Start LLM Answer<<<" in line):
                answer.append(line)
            if(">>>End LLM Answer<<<" in line):
                answer.pop(0)
                answers.append(answer)
                answer = []
            if(">>>Start LLM Answer<<<" not in line and ">>>End LLM Answer<<<" not in line):
                answer.append(line)
        idSnippet = file.split("conf")[0]
        PET = file.split("conf")[1].split("_")[0][1:]
        configuration = "conf"+file.split("conf")[1].split("_")[0][:1]
        fileAnswers[file] = [idSnippet, PET, configuration, answers]
#    for answersPerFile in fileAnswers:
#        print(answersPerFile)
#        for answers in fileAnswers[answersPerFile]:
#            print(answers)
    #print(len(fileAnswers))
    return fileAnswers

def comparison(gtFile,fileAnswers):
    f = open(gtFile)
    data = json.load(f)
    df = pd.DataFrame(columns=["ID", "PET", "Configuration", "Score", "Length", "Exact Match"])
    for answersPerFile in fileAnswers:
        if(isinstance(answersPerFile, str)):
            id = answersPerFile.split("conf")[0]
        for res in data:
            if (str(res['id'])==id):
                input = res['input'].split(";")[-1]
                gt = res['gt']
                gt = gt.replace(' . ', '.').replace(" ;", ";").replace("<s>", "")
                #print(gt)

        for answers in fileAnswers[answersPerFile][3]:
            conf = fileAnswers[answersPerFile][2]
            pet = answersPerFile.split(conf)[1].replace("_answer.txt", "")
            if(conf=="conf0"):
                score, exact = compareShort(gt, answers[0].rstrip().replace("`", ""),conf)
                length = len(answers[0].rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "SystemPrompt-NoCustomTag", score, length, exact]
            elif(conf=="conf1"):
                score, exact = compareShort(gt, answers[0].rstrip().replace("`", ""),conf)
                length = len(answers[0].rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "SystemPrompt-CustomTagNotExplained", score, length, exact]
            elif(conf=="conf2"):
                score, exact = compareShort(gt, answers[0].rstrip().replace("`", "").replace("<code> ","").replace("</code>",""),conf)
                length = len(answers[0].rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "SystemPrompt-CustomTagExplainedInPrompt", score, length, exact]
            elif(conf=="conf3"):
                score, exact = compareShort(gt, answers[0].rstrip().replace("`", "").replace("<code> ","").replace("</code>",""),conf)
                length = len(answers[0].rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "SystemPrompt-CustomTagExplainedInSystem", score, length, exact]
            elif(conf=="conf4"):
                score, exact = compareShort(gt, answers[0].rstrip().replace("`", "").replace("<code> ","").replace("</code>",""),conf)
                length = len(answers[0].rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "NoSystemPrompt", score, length, exact]

    df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values="Exact Match", aggfunc="sum")
    df_grouped = df_pivoted.groupby("Configuration").sum()#.mean()
    print(df_grouped)
def compareShort(gt,answer,conf):
    score = nltk.edit_distance(gt.replace(" ",""), answer.replace(" ",""))
    exact_match_counter = 0
    #if score < 2 and conf == "conf5":
    if score <= 2:
        exact_match_counter += 1
        #print("Ground Truth: "+gt)
        #print("LLM Answer: "+answer)
    #print("Edit Distance: "+str(score))
    return score, exact_match_counter

    #        print(answers)

#loadAnswers("../Answers/AnswersFewShots")
comparison("../GroundTruth/Updated_Test.json", loadAnswers("../Experiments/Answers/AnswersZeroShot"))
comparison("../GroundTruth/Updated_Test.json", loadAnswers("../Experiments/Answers/AnswersOneShot"))
comparison("../GroundTruth/Updated_Test.json", loadAnswers("../Experiments/Answers/AnswersFewShots"))
