import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,pipeline
from codecarbon import EmissionsTracker
import time
import json
import os

def loadModel():

	TOKEN=""
	#model_id = "meta-llama/Meta-Llama-3-8B"
	model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


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

def fewShots(Snippets,ids,model,tokenizer,AnswersDir,QuestionsDir,ExperimentsDir,limitSnippets):
	
	few_shots_messages_template_base = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in code completion for Java. Your task is to complete the provided Java code segment with one line. Give only the code completion.",
    },
		
	{
		'role': 'user', 
		'content': 'Example of code to be completed: package com.asakusafw.windgate.hadoopfs.ssh; import java.io.IOException; import java.io.InputStream; import java.util.zip.ZipInputStream; public class ZipEntryInputStream extends InputStream{ private final ZipInputStream zipped; private boolean closed = false; public ZipEntryInputStream(ZipInputStream zipped){ if(zipped == null){ throw new IllegalArgumentException(""<STR_LIT>"");} this.zipped = zipped;} @Override public void close() throws IOException{ if(closed == false){ zipped.closeEntry();} closed = true;} @Override public int read(byte[] b) throws IOException{ return zipped.read(b);} @Override public int read() throws IOException{ return zipped.read();} @Override public int available() throws IOException{ return zipped.available();} @Override public int read('
	},
		
	{
		'role': 'assistant', 
		'content': 'byte [ ] b , int off , int len ) throws IOException'
	},
		
	{	'role': 'user',
	 	'content': 'Example of code to be completed: package com.lmax.disruptor.support; import java.util.concurrent.ThreadFactory; public final'},
			
	{	'role': 'assistant', 
	 	'content': 'class DaemonThreadFactory implements ThreadFactory'
	},
			
	{	'role': 'user', 
	 	'content': 'Example of code to be completed: package org.rubypeople.rdt.refactoring.ui.pages.extractmethod; import org.eclipse.osgi.util.NLS; public class Messages extends NLS{ private static final String BUNDLE_NAME = "<STR_LIT>"; public static String ExtractMethodComposite_AccessModifier; public static String ExtractMethodComposite_ButtonDown; public static String ExtractMethodComposite_ButtonEdit; public static String ExtractMethodComposite_ButtonUp; public static String ExtractMethodComposite_ExpansionHint; public static String ExtractMethodComposite_MethodName; public static String ExtractMethodComposite_Name; public static String ExtractMethodComposite_Parameters; public static String ExtractMethodComposite_ReplaceAll; public static String ExtractMethodComposite_SameAsSource; public static String ExtractMethodComposite_SelectedCode; public static String ExtractMethodComposite_SignaturePreview; public static String MethodNameListener_IsNotValidName;'
	},
	
	{
		'role': 'assistant', 
		'content': 'public static String ParametersTableCellEditorListener_CannotHaveParametersWithEqualNames ;'
	},
			
	{	
		'role': 'user', 
		'content': 'Example of code to be completed: package com.asakusafw.windgate.file.resource; import static org.hamcrest.CoreMatchers.*; import static org.junit.Assert.*; import java.io.File; import java.io.FileOutputStream; import java.io.IOException; import java.io.ObjectOutputStream; import java.util.Arrays; import java.util.Collections; import org.junit.Rule; import org.junit.Test; import org.junit.rules.TemporaryFolder; import com.asakusafw.windgate.core.DriverScript; import com.asakusafw.windgate.core.GateScript; import com.asakusafw.windgate.core.ProcessScript; import com.asakusafw.windgate.core.resource.DrainDriver; import com.asakusafw.windgate.core.resource.SourceDriver; import com.asakusafw.windgate.core.vocabulary.FileProcess; public class FileResourceMirrorTest{ @Rule public TemporaryFolder folder = new TemporaryFolder(); @Test public void getName() throws Exception{ FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ assertThat(resource.getName(), is("<STR_LIT>"));} finally{ resource.close();}} @Test public void prepare() throws Exception{ File source = folder.newFile("<STR_LIT:source>"); File drain = folder.newFile("<STR_LIT>"); FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ ProcessScript<String> script = script(source, drain); resource.prepare(gate(script));} finally{ resource.close();}} @Test public void createSource'},
	
	{	
		'role': 'assistant', 
		'content': '( ) throws Exception'
	},
			
	{	
		'role': 'user', 
		'content': 'Example of code to be completed: package org.rubypeople.rdt.internal.debug.ui; import java.util.Hashtable; import org.eclipse.core.resources.IWorkspace; import org.eclipse.core.runtime.IConfigurationElement; import org.eclipse.core.runtime.IExtension; import org.eclipse.core.runtime.IExtensionPoint; import org.eclipse.core.runtime.IProgressMonitor; import org.eclipse.core.runtime.IStatus; import org.eclipse.core.runtime.Platform; import org.eclipse.core.runtime.Status; import org.eclipse.core.runtime.jobs.Job; import org.eclipse.jface.dialogs.ErrorDialog; import org.eclipse.jface.dialogs.MessageDialog; import org.eclipse.jface.resource.ImageDescriptor; import org.eclipse.swt.graphics.Image; import org.eclipse.swt.widgets.Display; import org.eclipse.swt.widgets.Shell; import org.eclipse.ui.IWorkbenchPage; import org.eclipse.ui.IWorkbenchWindow; import org.eclipse.ui.plugin.AbstractUIPlugin; import org.osgi.framework.BundleContext; import org.rubypeople.rdt.core.RubyCore; import org.rubypeople.rdt.debug.ui.IEvaluationContextManager; import org.rubypeople.rdt.debug.ui.RdtDebugUiConstants; import org.rubypeople.rdt.internal.debug.core.model.RubyVariable; import org.rubypeople.rdt.internal.debug.ui.evaluation.EvaluationExpressionModel; import org.rubypeople.rdt.ui.PreferenceConstants; import org.rubypeople.rdt.ui.text.RubyTextTools; import org.rubypeople.'
	},
	
	{	
		'role': 'assistant', 
		'content': 'rdt . ui . viewsupport . ImageDescriptorRegistry ;'
	},
		
    {
        "role": "user",
        "content": None
    },
	]
	
	few_shots_messages_template_embedded = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in code completion for Java. Your task is to complete the provided Java code segment with one line. Give only the code completion.",
    },
		
	{
		'role': 'user', 
		'content': 'Example of code to be completed: <code> package com.asakusafw.windgate.hadoopfs.ssh; import java.io.IOException; import java.io.InputStream; import java.util.zip.ZipInputStream; public class ZipEntryInputStream extends InputStream{ private final ZipInputStream zipped; private boolean closed = false; public ZipEntryInputStream(ZipInputStream zipped){ if(zipped == null){ throw new IllegalArgumentException(""<STR_LIT>"");} this.zipped = zipped;} @Override public void close() throws IOException{ if(closed == false){ zipped.closeEntry();} closed = true;} @Override public int read(byte[] b) throws IOException{ return zipped.read(b);} @Override public int read() throws IOException{ return zipped.read();} @Override public int available() throws IOException{ return zipped.available();}</code> <incomplete> @Override public int read( </incomplete>'
	},
		
	{
		'role': 'assistant', 
		'content': 'byte [ ] b , int off , int len ) throws IOException'
	},
		
	{	'role': 'user',
	 	'content': 'Example of code to be completed: <code> package com.lmax.disruptor.support; import java.util.concurrent.ThreadFactory;</code> <incomplete> public final </incomplete>'
	},
			
	{	'role': 'assistant', 
	 	'content': 'class DaemonThreadFactory implements ThreadFactory'
	},
			
	{	'role': 'user', 
	 	'content': 'Example of code to be completed: <code> package org.rubypeople.rdt.refactoring.ui.pages.extractmethod; import org.eclipse.osgi.util.NLS; public class Messages extends NLS{ private static final String BUNDLE_NAME = "<STR_LIT>"; public static String ExtractMethodComposite_AccessModifier; public static String ExtractMethodComposite_ButtonDown; public static String ExtractMethodComposite_ButtonEdit; public static String ExtractMethodComposite_ButtonUp; public static String ExtractMethodComposite_ExpansionHint; public static String ExtractMethodComposite_MethodName; public static String ExtractMethodComposite_Name; public static String ExtractMethodComposite_Parameters; public static String ExtractMethodComposite_ReplaceAll; public static String ExtractMethodComposite_SameAsSource; public static String ExtractMethodComposite_SelectedCode; public static String ExtractMethodComposite_SignaturePreview;</code> <incomplete> public static String MethodNameListener_IsNotValidName; </incomplete>'
	},
	
	{
		'role': 'assistant', 
		'content': 'public static String ParametersTableCellEditorListener_CannotHaveParametersWithEqualNames ;'
	},
			
	{	
		'role': 'user', 
		'content': 'Example of code to be completed: <code> package com.asakusafw.windgate.file.resource; import static org.hamcrest.CoreMatchers.*; import static org.junit.Assert.*; import java.io.File; import java.io.FileOutputStream; import java.io.IOException; import java.io.ObjectOutputStream; import java.util.Arrays; import java.util.Collections; import org.junit.Rule; import org.junit.Test; import org.junit.rules.TemporaryFolder; import com.asakusafw.windgate.core.DriverScript; import com.asakusafw.windgate.core.GateScript; import com.asakusafw.windgate.core.ProcessScript; import com.asakusafw.windgate.core.resource.DrainDriver; import com.asakusafw.windgate.core.resource.SourceDriver; import com.asakusafw.windgate.core.vocabulary.FileProcess; public class FileResourceMirrorTest{ @Rule public TemporaryFolder folder = new TemporaryFolder(); @Test public void getName() throws Exception{ FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ assertThat(resource.getName(), is("<STR_LIT>"));} finally{ resource.close();}} @Test public void prepare() throws Exception{ File source = folder.newFile("<STR_LIT:source>"); File drain = folder.newFile("<STR_LIT>"); FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ ProcessScript<String> script = script(source, drain); resource.prepare(gate(script));} finally{ resource.close();}}</code> <incomplete> @Test public void createSource </incomplete>'
	},
	
	{	
		'role': 'assistant', 
		'content': '( ) throws Exception'
	},
			
	{	
		'role': 'user', 
		'content': 'Example of code to be completed: <code> package org.rubypeople.rdt.internal.debug.ui; import java.util.Hashtable; import org.eclipse.core.resources.IWorkspace; import org.eclipse.core.runtime.IConfigurationElement; import org.eclipse.core.runtime.IExtension; import org.eclipse.core.runtime.IExtensionPoint; import org.eclipse.core.runtime.IProgressMonitor; import org.eclipse.core.runtime.IStatus; import org.eclipse.core.runtime.Platform; import org.eclipse.core.runtime.Status; import org.eclipse.core.runtime.jobs.Job; import org.eclipse.jface.dialogs.ErrorDialog; import org.eclipse.jface.dialogs.MessageDialog; import org.eclipse.jface.resource.ImageDescriptor; import org.eclipse.swt.graphics.Image; import org.eclipse.swt.widgets.Display; import org.eclipse.swt.widgets.Shell; import org.eclipse.ui.IWorkbenchPage; import org.eclipse.ui.IWorkbenchWindow; import org.eclipse.ui.plugin.AbstractUIPlugin; import org.osgi.framework.BundleContext; import org.rubypeople.rdt.core.RubyCore; import org.rubypeople.rdt.debug.ui.IEvaluationContextManager; import org.rubypeople.rdt.debug.ui.RdtDebugUiConstants; import org.rubypeople.rdt.internal.debug.core.model.RubyVariable; import org.rubypeople.rdt.internal.debug.ui.evaluation.EvaluationExpressionModel; import org.rubypeople.rdt.ui.PreferenceConstants; import org.rubypeople.rdt.ui.text.RubyTextTools;</code> <incomplete> import org.rubypeople. </incomplete>'
	},
	
	{	
		'role': 'assistant', 
		'content': 'rdt . ui . viewsupport . ImageDescriptorRegistry ;'
	},
		
    {
        "role": "user",
        "content": None
    },
	]
	
	few_shots_messages_template_embedded_explained = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in code completion for Java. Your task is to complete the provided Java code segment with one line. Give only the code completion. The <code> tag marks the code to analyze, the <incomplete> tag marks the line to be completed.",
    },
		
	{
		'role': 'user', 
		'content': 'Example of code to be completed: <code> package com.asakusafw.windgate.hadoopfs.ssh; import java.io.IOException; import java.io.InputStream; import java.util.zip.ZipInputStream; public class ZipEntryInputStream extends InputStream{ private final ZipInputStream zipped; private boolean closed = false; public ZipEntryInputStream(ZipInputStream zipped){ if(zipped == null){ throw new IllegalArgumentException(""<STR_LIT>"");} this.zipped = zipped;} @Override public void close() throws IOException{ if(closed == false){ zipped.closeEntry();} closed = true;} @Override public int read(byte[] b) throws IOException{ return zipped.read(b);} @Override public int read() throws IOException{ return zipped.read();} @Override public int available() throws IOException{ return zipped.available();}</code> <incomplete> @Override public int read( </incomplete>'
	},
		
	{
		'role': 'assistant', 
		'content': 'byte [ ] b , int off , int len ) throws IOException'
	},
		
	{	'role': 'user',
	 	'content': 'Example of code to be completed: <code> package com.lmax.disruptor.support; import java.util.concurrent.ThreadFactory;</code> <incomplete> public final </incomplete>'
	},
			
	{	'role': 'assistant', 
	 	'content': 'class DaemonThreadFactory implements ThreadFactory'
	},
			
	{	'role': 'user', 
	 	'content': 'Example of code to be completed: <code> package org.rubypeople.rdt.refactoring.ui.pages.extractmethod; import org.eclipse.osgi.util.NLS; public class Messages extends NLS{ private static final String BUNDLE_NAME = "<STR_LIT>"; public static String ExtractMethodComposite_AccessModifier; public static String ExtractMethodComposite_ButtonDown; public static String ExtractMethodComposite_ButtonEdit; public static String ExtractMethodComposite_ButtonUp; public static String ExtractMethodComposite_ExpansionHint; public static String ExtractMethodComposite_MethodName; public static String ExtractMethodComposite_Name; public static String ExtractMethodComposite_Parameters; public static String ExtractMethodComposite_ReplaceAll; public static String ExtractMethodComposite_SameAsSource; public static String ExtractMethodComposite_SelectedCode; public static String ExtractMethodComposite_SignaturePreview;</code> <incomplete> public static String MethodNameListener_IsNotValidName; </incomplete>'
	},
	
	{
		'role': 'assistant', 
		'content': 'public static String ParametersTableCellEditorListener_CannotHaveParametersWithEqualNames ;'
	},
			
	{	
		'role': 'user', 
		'content': 'Example of code to be completed: <code> package com.asakusafw.windgate.file.resource; import static org.hamcrest.CoreMatchers.*; import static org.junit.Assert.*; import java.io.File; import java.io.FileOutputStream; import java.io.IOException; import java.io.ObjectOutputStream; import java.util.Arrays; import java.util.Collections; import org.junit.Rule; import org.junit.Test; import org.junit.rules.TemporaryFolder; import com.asakusafw.windgate.core.DriverScript; import com.asakusafw.windgate.core.GateScript; import com.asakusafw.windgate.core.ProcessScript; import com.asakusafw.windgate.core.resource.DrainDriver; import com.asakusafw.windgate.core.resource.SourceDriver; import com.asakusafw.windgate.core.vocabulary.FileProcess; public class FileResourceMirrorTest{ @Rule public TemporaryFolder folder = new TemporaryFolder(); @Test public void getName() throws Exception{ FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ assertThat(resource.getName(), is("<STR_LIT>"));} finally{ resource.close();}} @Test public void prepare() throws Exception{ File source = folder.newFile("<STR_LIT:source>"); File drain = folder.newFile("<STR_LIT>"); FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ ProcessScript<String> script = script(source, drain); resource.prepare(gate(script));} finally{ resource.close();}}</code> <incomplete> @Test public void createSource </incomplete>'
	},
	
	{	
		'role': 'assistant', 
		'content': '( ) throws Exception'
	},
			
	{	
		'role': 'user', 
		'content': 'Example of code to be completed: <code> package org.rubypeople.rdt.internal.debug.ui; import java.util.Hashtable; import org.eclipse.core.resources.IWorkspace; import org.eclipse.core.runtime.IConfigurationElement; import org.eclipse.core.runtime.IExtension; import org.eclipse.core.runtime.IExtensionPoint; import org.eclipse.core.runtime.IProgressMonitor; import org.eclipse.core.runtime.IStatus; import org.eclipse.core.runtime.Platform; import org.eclipse.core.runtime.Status; import org.eclipse.core.runtime.jobs.Job; import org.eclipse.jface.dialogs.ErrorDialog; import org.eclipse.jface.dialogs.MessageDialog; import org.eclipse.jface.resource.ImageDescriptor; import org.eclipse.swt.graphics.Image; import org.eclipse.swt.widgets.Display; import org.eclipse.swt.widgets.Shell; import org.eclipse.ui.IWorkbenchPage; import org.eclipse.ui.IWorkbenchWindow; import org.eclipse.ui.plugin.AbstractUIPlugin; import org.osgi.framework.BundleContext; import org.rubypeople.rdt.core.RubyCore; import org.rubypeople.rdt.debug.ui.IEvaluationContextManager; import org.rubypeople.rdt.debug.ui.RdtDebugUiConstants; import org.rubypeople.rdt.internal.debug.core.model.RubyVariable; import org.rubypeople.rdt.internal.debug.ui.evaluation.EvaluationExpressionModel; import org.rubypeople.rdt.ui.PreferenceConstants; import org.rubypeople.rdt.ui.text.RubyTextTools;</code> <incomplete> import org.rubypeople. </incomplete>'
	},
	
	{	
		'role': 'assistant', 
		'content': 'rdt . ui . viewsupport . ImageDescriptorRegistry ;'
	},
		
    {
        "role": "user",
        "content": None
    },
	]
	
	few_shots_messages_template_empty = [
    {
        "role": "system",
        "content": "",
    },
		
	{
		'role': 'user', 
		'content': 'package com.asakusafw.windgate.hadoopfs.ssh; import java.io.IOException; import java.io.InputStream; import java.util.zip.ZipInputStream; public class ZipEntryInputStream extends InputStream{ private final ZipInputStream zipped; private boolean closed = false; public ZipEntryInputStream(ZipInputStream zipped){ if(zipped == null){ throw new IllegalArgumentException(""<STR_LIT>"");} this.zipped = zipped;} @Override public void close() throws IOException{ if(closed == false){ zipped.closeEntry();} closed = true;} @Override public int read(byte[] b) throws IOException{ return zipped.read(b);} @Override public int read() throws IOException{ return zipped.read();} @Override public int available() throws IOException{ return zipped.available();} @Override public int read('
	},
		
	{
		'role': 'assistant', 
		'content': 'byte [ ] b , int off , int len ) throws IOException'
	},
		
	{	'role': 'user',
	 	'content': 'package com.lmax.disruptor.support; import java.util.concurrent.ThreadFactory; public final'},
			
	{	'role': 'assistant', 
	 	'content': 'class DaemonThreadFactory implements ThreadFactory'
	},
			
	{	'role': 'user', 
	 	'content': 'package org.rubypeople.rdt.refactoring.ui.pages.extractmethod; import org.eclipse.osgi.util.NLS; public class Messages extends NLS{ private static final String BUNDLE_NAME = "<STR_LIT>"; public static String ExtractMethodComposite_AccessModifier; public static String ExtractMethodComposite_ButtonDown; public static String ExtractMethodComposite_ButtonEdit; public static String ExtractMethodComposite_ButtonUp; public static String ExtractMethodComposite_ExpansionHint; public static String ExtractMethodComposite_MethodName; public static String ExtractMethodComposite_Name; public static String ExtractMethodComposite_Parameters; public static String ExtractMethodComposite_ReplaceAll; public static String ExtractMethodComposite_SameAsSource; public static String ExtractMethodComposite_SelectedCode; public static String ExtractMethodComposite_SignaturePreview; public static String MethodNameListener_IsNotValidName;'
	},
	
	{
		'role': 'assistant', 
		'content': 'public static String ParametersTableCellEditorListener_CannotHaveParametersWithEqualNames ;'
	},
			
	{	
		'role': 'user', 'content': 'package com.asakusafw.windgate.file.resource; import static org.hamcrest.CoreMatchers.*; import static org.junit.Assert.*; import java.io.File; import java.io.FileOutputStream; import java.io.IOException; import java.io.ObjectOutputStream; import java.util.Arrays; import java.util.Collections; import org.junit.Rule; import org.junit.Test; import org.junit.rules.TemporaryFolder; import com.asakusafw.windgate.core.DriverScript; import com.asakusafw.windgate.core.GateScript; import com.asakusafw.windgate.core.ProcessScript; import com.asakusafw.windgate.core.resource.DrainDriver; import com.asakusafw.windgate.core.resource.SourceDriver; import com.asakusafw.windgate.core.vocabulary.FileProcess; public class FileResourceMirrorTest{ @Rule public TemporaryFolder folder = new TemporaryFolder(); @Test public void getName() throws Exception{ FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ assertThat(resource.getName(), is("<STR_LIT>"));} finally{ resource.close();}} @Test public void prepare() throws Exception{ File source = folder.newFile("<STR_LIT:source>"); File drain = folder.newFile("<STR_LIT>"); FileResourceMirror resource = new FileResourceMirror("<STR_LIT>"); try{ ProcessScript<String> script = script(source, drain); resource.prepare(gate(script));} finally{ resource.close();}} @Test public void createSource'},
	
	{	
		'role': 'assistant', 
		'content': '( ) throws Exception'
	},
			
	{	
		'role': 'user', 
		'content': 'package org.rubypeople.rdt.internal.debug.ui; import java.util.Hashtable; import org.eclipse.core.resources.IWorkspace; import org.eclipse.core.runtime.IConfigurationElement; import org.eclipse.core.runtime.IExtension; import org.eclipse.core.runtime.IExtensionPoint; import org.eclipse.core.runtime.IProgressMonitor; import org.eclipse.core.runtime.IStatus; import org.eclipse.core.runtime.Platform; import org.eclipse.core.runtime.Status; import org.eclipse.core.runtime.jobs.Job; import org.eclipse.jface.dialogs.ErrorDialog; import org.eclipse.jface.dialogs.MessageDialog; import org.eclipse.jface.resource.ImageDescriptor; import org.eclipse.swt.graphics.Image; import org.eclipse.swt.widgets.Display; import org.eclipse.swt.widgets.Shell; import org.eclipse.ui.IWorkbenchPage; import org.eclipse.ui.IWorkbenchWindow; import org.eclipse.ui.plugin.AbstractUIPlugin; import org.osgi.framework.BundleContext; import org.rubypeople.rdt.core.RubyCore; import org.rubypeople.rdt.debug.ui.IEvaluationContextManager; import org.rubypeople.rdt.debug.ui.RdtDebugUiConstants; import org.rubypeople.rdt.internal.debug.core.model.RubyVariable; import org.rubypeople.rdt.internal.debug.ui.evaluation.EvaluationExpressionModel; import org.rubypeople.rdt.ui.PreferenceConstants; import org.rubypeople.rdt.ui.text.RubyTextTools; import org.rubypeople.'
	},
	
	{	
		'role': 'assistant', 
		'content': 'rdt . ui . viewsupport . ImageDescriptorRegistry ;'
	},
		
    {
        "role": "user",
        "content": None
    },
	]
		
	Request_One_Line = "Complete the snippet adding one line please."
	Custom_Snippet_Explained_With_Marks = "The code to analyze is embedded in the <code> tag and the line to be completed is embedded in the <incomplete> tag."
	
	configurations = []
	
	Snippets = Snippets[:limitSnippets]
	for snippet, id in zip(Snippets,ids):
		
		last_position = snippet.rfind(";")
		reduced_snippet = snippet[:last_position+1]
		uncompleted_line = snippet[last_position+1:]
		

		#CONF1
		few_shots_messages_template_base[-1]['content'] = snippet
		system_conf1 = few_shots_messages_template_base[0:len(few_shots_messages_template_base)-1]
		message_conf1 = few_shots_messages_template_base[-1]['content']
		input_ids_conf1 = tokenizer.apply_chat_template(few_shots_messages_template_base,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF2
		few_shots_messages_template_embedded[-1]['content'] = "<code> " + reduced_snippet + " </code>" + "<incomplete>" + uncompleted_line + "</incomplete>"
		system_conf2 = few_shots_messages_template_embedded[0:len(few_shots_messages_template_embedded)-1]
		message_conf2 = few_shots_messages_template_embedded[-1]['content']
		input_ids_conf2 = tokenizer.apply_chat_template(few_shots_messages_template_embedded,add_generation_prompt=True,return_tensors="pt").to(model.device)	
		
		#CONF3
		few_shots_messages_template_embedded[-1]['content'] = Custom_Snippet_Explained_With_Marks + "<code> " + reduced_snippet + " </code>" + "<incomplete>" + uncompleted_line + "</incomplete>"
		system_conf3 = few_shots_messages_template_embedded[0:len(few_shots_messages_template_embedded)-1]
		message_conf3 = few_shots_messages_template_embedded[-1]['content']
		input_ids_conf3 = tokenizer.apply_chat_template(few_shots_messages_template_embedded,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF4
		few_shots_messages_template_embedded_explained[-1]['content'] = "<code> " + reduced_snippet + " </code>" + "<incomplete>" + uncompleted_line + "</incomplete>"
		system_conf4 = few_shots_messages_template_embedded_explained[0:len(few_shots_messages_template_embedded_explained)-1]
		message_conf4 = few_shots_messages_template_embedded_explained[-1]['content']
		input_ids_conf4 = tokenizer.apply_chat_template(few_shots_messages_template_embedded_explained,add_generation_prompt=True,return_tensors="pt").to(model.device)	
		
		#CONF5
		few_shots_messages_template_empty[-1]['content'] = Request_One_Line + snippet
		system_conf5 = few_shots_messages_template_empty[0:len(few_shots_messages_template_empty)-1]
		message_conf5 = few_shots_messages_template_empty[-1]['content']
		input_ids_conf5 = tokenizer.apply_chat_template(few_shots_messages_template_empty,add_generation_prompt=True,return_tensors="pt").to(model.device)
	
	
		configurations = [[input_ids_conf1,"conf1",system_conf1,message_conf1],[input_ids_conf2,"conf2",system_conf2,message_conf2],[input_ids_conf3,"conf3",system_conf3,message_conf3],[input_ids_conf4,"conf4",system_conf4,message_conf4],[input_ids_conf5,"conf5",system_conf5,message_conf5]]

	
		for input_ids in configurations:
			writerAnswers = open(AnswersDir+"/"+str(id)+input_ids[1]+"fewShots_answer.txt","a")
			writerQuestions = open(QuestionsDir+"/"+str(id)+input_ids[1]+"fewShots_question.txt","a")
			for i in range(0,5):

				tracker=EmissionsTracker(measure_power_secs=0.1, tracking_mode="process",output_file=ExperimentsDir+"/"+str(id)+input_ids[1]+"fewShots.csv")
				tracker.start()
				
				tokenized_chat = torch.tensor(input_ids[0][0])
				tokenized_chat = tokenized_chat.unsqueeze(0)
				tokenized_chat = tokenized_chat.to(model.device)
				generated_text = model.generate(tokenized_chat, max_new_tokens=512)

				generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

				tracker.stop()
				
				writerQuestions.write(">>>Start LLM System<<<\n")
				for elem in input_ids[2]:
					writerQuestions.write('role:'+elem['role']+"\n")
					writerQuestions.write('content:'+elem['content']+"\n")
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
questionsFolder = baseFolder+"AnswersFewShots"
answersFolder = baseFolder+"QuestionsFewShots"
measurementsFolder = baseFolder+"ExperimentsFewShots"
createFolders(questionsFolder,answersFolder,measurementsFolder)
snippetsDone = recoverState(measurementsFolder)
Snippets, ids = loadConfigurations(measurementsFolder,AnswersDirs,snippetsDone)
model,tokenizer = loadModel()
fewShots(Snippets,ids,model,tokenizer,questionsFolder,answersFolder, measurementsFolder, limitSnippets)


print('Emissions tracking llama completed')


