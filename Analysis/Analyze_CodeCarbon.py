import pandas as pd
import os
base_directory = "../Experiments/CodeCarbon/Experiments"

def createFile(base_directory,PET):
    confs = {
        "0": "SystemPrompt-NoCustomTag",
        "1": "SystemPrompt-CustomTagNotExplained",
        "2": "SystemPrompt-CustomTagExplainedInPrompt",
        "3": "SystemPrompt-CustomTagExplainedInSystem",
        "4": "NoSystemPrompt",
    }

    results = {}
    results_list = []
    resultData = []
    for pet in PET:
        filesDir = base_directory+pet
        for file in os.listdir(filesDir):
            conf = file.split("conf")
            idExperiment = conf[0]
            conf = conf[1].lower().split(".")[0].replace(pet.lower(),"")
            data = pd.read_csv(filesDir+"/"+file)
            average_consumption = data['gpu_energy'].mean()
            average_time = data['duration'].mean()
            #print(file+":"+str(average_consumption))
            resultData.append(idExperiment +":"+pet+":"+ confs[conf] +":"+ str(average_consumption)+":"+str(average_time))
    return resultData

def calculateAverage(base_directory):

    df = pd.read_csv(base_directory+'/data.csv', sep=';')
    df['consumption'] = pd.to_numeric(df['consumption'].str.replace(',', '.'))#, errors='coerce'
    result = df.groupby('configuration')['consumption'].mean()

def processEnergy(data):
    df = pd.DataFrame([s.split(":") for s in data], columns=["ID", "PET", "Configuration", "EnergyValue", "Duration"])
    df.update(df.apply(pd.to_numeric, downcast='float', errors='coerce'))
    df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values="EnergyValue", aggfunc="first")
    df_pivoted.columns = ["FewShot", "OneShot", "ZeroShot"]
    #print(df_pivoted)
    df_grouped = df_pivoted.groupby("Configuration").mean()
    print(df_grouped)
    #df_averages = df_grouped.mean().reset_index()
    #print(df_averages)

def processDuration(data):
    df = pd.DataFrame([s.split(":") for s in data], columns=["ID", "PET", "Configuration", "EnergyValue", "Duration"])
    df.update(df.apply(pd.to_numeric, errors='coerce'))
    df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values="Duration", aggfunc="first")
    df_pivoted.columns = ["FewShot", "OneShot", "ZeroShot"]
    #print(df_pivoted)
    df_grouped = df_pivoted.groupby("Configuration").mean()
    print(df_grouped)

PET = ["ZeroShot", "OneShot", "FewShots"]
#PET = ["ZeroShot"]
#calculateAverage(base_directory)
data = createFile(base_directory, PET)

processEnergy(data)
processDuration(data)
