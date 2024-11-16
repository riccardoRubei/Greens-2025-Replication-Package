import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,pipeline
from codecarbon import EmissionsTracker
import time
import json
import os

def loadModel():

	TOKEN=""
	model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
	#model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


	print('Loading Tokenizer')
	tokenizer = AutoTokenizer.from_pretrained(model_id,token=TOKEN)
	tokenizer.pad_token = tokenizer.eos_token

	nf4_config = BitsAndBytesConfig(
	   load_in_4bit=True,
	   bnb_4bit_quant_type="nf4",
	   bnb_4bit_use_double_quant=True,
	   bnb_4bit_compute_dtype=torch.bfloat16
	)

	terminators = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids("<|eot_id|>")
	]
		
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
    	torch_dtype=torch.bfloat16,
    	device_map="auto",
    	quantization_config=nf4_config,
    	token=TOKEN
	)

	print('Model loaded')
	return model, tokenizer

	
	
ExperimentsDir = "Experiments"
AnswersDirs = "Answers"
def loadConfigurations(ExperimentsDir,AnswersDirs,snippetsDone):

	Snippets = []
	ids = []
	maxLength = 2000
	#problems if length > 3922
	#maxLength = 100
	minLength = 1900
	jsonFile = open("Updated_Test.json")
	data = json.load(jsonFile)

	for entry in data:
		if len(entry['input'])<maxLength:
		#if len(entry['input'])>minLength and len(entry['input'])<2000:
			#Snippets.append(entry['input'])
			if(str(entry['id']) not in snippetsDone):
				Snippets.append(entry['input'].replace("<s> ","").replace("</s>","").replace(" . ",".").replace(" .",".").replace(" ;",";"))
				ids.append(entry['id'])
	print(len(Snippets))
	return Snippets, ids


######TEST SPECIFIC SNIPPET######
#Snippets = []
#ids = []
#for entry in data:
#	if(entry['id'] == 127):
#		print(len(entry['input']))
#		Snippets.append(entry['input'].replace("<s> ","").replace("</s>","").replace(". ","").replace(" ;",";"))
#		ids.append(entry['id'])	

		
def recoverState(ExperimentsDir):
	snippetsDone = []
	for file in os.listdir(ExperimentsDir):
		id = file.split("conf")[0]
		snippetsDone.append(id)
	return snippetsDone

def zeroShot(Snippets,ids,model,tokenizer,AnswersDir,QuestionsDir,ExperimentsDir,limitSnippets):
	
	zero_shot_messages_template = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in code completion for Java. Your task is to complete the provided Java code segment with one line. Give only the code completion.",
    },
    {
        "role": "user",
        "content": None
    }
	]
	
	zero_shot_messages_template_empty = [
    {
        "role": "system",
        "content": "",
    },
    {
        "role": "user",
        "content": None
    }
	]
	
	zero_shot_messages_template_Conf4 = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in code completion for Java. Your task is to complete the provided Java code segment with one line. Give only the code completion. The code to analyze is marked by the <code> tag and the line to be completed is marked by the <incomplete> tag.",
    },
    {
        "role": "user",
        "content": None
    }
	]
		
	Request_One_Line = "Hi, complete the following snippet adding one line please: "
	Custom_Snippet_Explained_With_Marks = "The code to analyze is marked by the <code> tag and the line to be completed is marked by the <incomplete> tag."
	configurations = []
	
	Snippets = Snippets[:limitSnippets]
	for snippet, id in zip(Snippets,ids):
		
		last_position = snippet.rfind(";")
		reduced_snippet = snippet[:last_position+1]
		uncompleted_line = snippet[last_position+1:]
		

		#CONF1
		zero_shot_messages_template[-1]['content'] = snippet
		system_conf1 = zero_shot_messages_template[0]['content']
		message_conf1 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf1 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF2
		zero_shot_messages_template[-1]['content'] = "<code> " + reduced_snippet + " </code>" + "<incomplete>" + uncompleted_line + "</incomplete>"
		system_conf2 = zero_shot_messages_template[0]['content']
		message_conf2 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf2 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)	
		
		#CONF3
		zero_shot_messages_template[-1]['content'] = Custom_Snippet_Explained_With_Marks + "<code> " + reduced_snippet + " </code>" + "<incomplete>" + uncompleted_line + "</incomplete>"
		system_conf3 = zero_shot_messages_template[0]['content']
		message_conf3 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf3 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF4
		zero_shot_messages_template_Conf4[-1]['content'] = "<code> " + reduced_snippet + " </code>" + "<incomplete>" + uncompleted_line + "</incomplete>"
		system_conf4 = zero_shot_messages_template_Conf4[0]['content']
		message_conf4 = zero_shot_messages_template_Conf4[-1]['content']
		
		input_ids_conf4 = tokenizer.apply_chat_template(zero_shot_messages_template_Conf4,add_generation_prompt=True,return_tensors="pt").to(model.device)	
		
		#CONF5
		zero_shot_messages_template_empty[-1]['content'] = Request_One_Line + snippet
		system_conf5 = zero_shot_messages_template_empty[0]['content']
		message_conf5 = zero_shot_messages_template_empty[-1]['content']
		
		input_ids_conf5 = tokenizer.apply_chat_template(zero_shot_messages_template_empty,add_generation_prompt=True,return_tensors="pt").to(model.device)
	
	
		configurations = [[input_ids_conf1,"conf1",system_conf1,message_conf1],[input_ids_conf2,"conf2",system_conf2,message_conf2],[input_ids_conf3,"conf3",system_conf3,message_conf3],[input_ids_conf4,"conf4",system_conf4,message_conf4],[input_ids_conf5,"conf5",system_conf5,message_conf5]]

	
		for input_ids in configurations:
			writerAnswers = open(AnswersDir+"/"+str(id)+input_ids[1]+"zeroShot_answer.txt","a")
			writerQuestions = open(QuestionsDir+"/"+str(id)+input_ids[1]+"zeroShot_question.txt","a")
			for i in range(0,5):

				tracker=EmissionsTracker(measure_power_secs=0.1, tracking_mode="process",output_file=ExperimentsDir+"/"+str(id)+input_ids[1]+"zeroShot.csv")
				tracker.start()
				
				tokenized_chat = torch.tensor(input_ids[0][0])
				tokenized_chat = tokenized_chat.unsqueeze(0)
				tokenized_chat = tokenized_chat.to(model.device)
				generated_text = model.generate(tokenized_chat, max_new_tokens=512)

				generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

				tracker.stop()
				
				writerQuestions.write(">>>Start LLM System<<<\n")
				writerQuestions.write(input_ids[2]+"\n")
				writerQuestions.write(">>>End LLM System<<<\n")
				writerQuestions.write(">>>Start LLM Prompt<<<\n")
				writerQuestions.write(input_ids[3]+"\n")
				writerQuestions.write(">>>End LLM Prompt<<<\n")
				
				if "assistant" in generated_text:
					generated_text = generated_text.split("assistant")[-1].strip()
					writerAnswers.write(">>>Start LLM Answer<<<\n")
					writerAnswers.write(generated_text+"\n")
					writerAnswers.write(">>>End LLM Answer<<<\n")
				else:
					writerAnswers.write(">>>Start LLM Answer<<<\n")
					writerAnswers.write(generated_text+"\n")
					writerAnswers.write(">>>End LLM Answer<<<\n")
				time.sleep(10)
	
def createFolders(Answers,Questions,Measurements):
	if not os.path.exists(Answers): 
		os.makedirs(Answers)
	if not os.path.exists(Questions): 
		os.makedirs(Questions) 
	if not os.path.exists(Measurements): 
		os.makedirs(Measurements) 

limitSnippets = 40
baseFolder = "14082024/"
questionsFolder = baseFolder+"AnswersZeroShot"
answersFolder = baseFolder+"QuestionsZeroShot"
measurementsFolder = baseFolder+"ExperimentsZeroShot"
createFolders(questionsFolder,answersFolder,measurementsFolder)
snippetsDone = recoverState(measurementsFolder)
Snippets, ids = loadConfigurations(measurementsFolder,AnswersDirs,snippetsDone)
model,tokenizer = loadModel()
zeroShot(Snippets,ids,model,tokenizer,questionsFolder,answersFolder, measurementsFolder, limitSnippets)

print('Emissions tracking llama completed')


