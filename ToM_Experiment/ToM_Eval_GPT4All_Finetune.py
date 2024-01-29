from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cot', action='store_true', default=False,
                    help='Add chain of thought examples to input prompt')
parser.add_argument('--sbs', action='store_true', default=False,
                    help='Add step-by-step reasoning instruction to input prompt')
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--reps', default=20, type=int)
parser.add_argument('--tokens', default=128, type=int)
parser.add_argument('--top_k', default=50, type=int)
parser.add_argument('--top_p', default=0.95, type=float)
parser.add_argument('--sample', action='store_true', default=True)
parser.add_argument('--temp', default=0.4, type=float)

def main():
    scenarios = json.load(open('tom_scenarios.json'))['scenarios']

    args = parser.parse_args()

    num_rep = args.reps
    # Model parameter
    max_new_tokens = args.tokens
    top_k = args.top_k
    do_sample = args.sample
    top_p = args.top_p
    temp = args.temp

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 device_map="auto",
                                                 load_in_8bit=True)

    parameter_dict = {
        'max_new_tokens': max_new_tokens,
        'top k': top_k,
        'top p': top_p,
        'do sample': do_sample,
        'temperature': temp
    }

    output_dict = {
        'model': args.model,
        'step-by_step reasoning': args.sbs,
        'chain-of-thougth examples': args.cot,
        'model parameter': parameter_dict,
        'scenarios': list()
    }
    
    # Create Chain-of-Thought examples
    if args.cot:
        cot_scenarios = json.load(open('cot_example_scenarios.json'))['scenarios']
        if args.sbs:
            cot_prompt = ''.join(["Context: ", cot_scenarios[0]['context'],
            ' Question: ', cot_scenarios[0]['question'],
            " Answer: Let's think step by step: ", cot_scenarios[0]['answer'], '\n\n',
            "Context: ", cot_scenarios[1]['context'],
            ' Question: ', cot_scenarios[1]['question'],
            " Answer: Let's think step by step: ", cot_scenarios[1]['answer'], '\n\n'])

        else:
            cot_prompt = ''.join(["Context: ", cot_scenarios[0]['context'],
            ' Question: ', cot_scenarios[0]['question'],
            ' Answer: ', cot_scenarios[0]['answer'], '\n\n',
            "Context: ", cot_scenarios[1]['context'],
            ' Question: ', cot_scenarios[1]['question'],
            ' Answer: ', cot_scenarios[1]['answer'], '\n\n'])


    for scenario in scenarios:
        scenario_id = scenario['id']
        print(f'\n> Scenario Nr.{scenario_id}')

        scenario_dict = {
            'scenario id': scenario_id,
            'answers': list()
        }

        for i in range(num_rep):
            print(f'\n> Repetition {i+1} of {num_rep}')
            
            # Create Base prompt with instruction, scenario and question prompt
            base_prompt = "Context: " + scenario['context'] + ' Question: ' + scenario['question'] + ' Answer: '
            
            # Add Step-by-Step prompting
            if args.sbs:
                base_prompt =  base_prompt +  "Let's think step by step: "

            if args.cot:
                prompt = "Answer the following question in context:\n\n" + cot_prompt + base_prompt
            else:
                prompt = "Answer the following question in context:\n\n" + base_prompt

            prompt_len = len(prompt)
            
            if True:
                # Tokenize base prompt
                inputs = tokenizer(prompt, return_tensors='pt').to('cuda').input_ids
                
                # Generate answer
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temp
                )
                
                # Decode output
                outputs = tokenizer.batch_decode(outputs)
            
            time_id = time.strftime('%H:%M:%S')

            scenario_dict['answers'].append(
                {
                    'repetition': i+1,
                    'output': outputs[0][prompt_len:],
                    'time': time_id
                 }
            )

        output_dict['scenarios'].append(scenario_dict)
    
    time_id = time.strftime('%H:%M:%S')
    info_string = time_id
    if args.sbs:
        info_string += '_sbs'
    if args.cot:
        info_string += '_cot'

    with open(os.path.join('ToM_Eval_GPT4All_Finetune', f'tom_output_{info_string}.json'), 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)

main()
