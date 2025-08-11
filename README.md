# MMMA
Multimodal-multiagent

Setup
1. conda create -n MMMA python=3.10 
2. pip install numpy
3. pip install google-genai
4. pip install datasets

Running:
(Note: Works for the MathVista Dataset)
1. python generate_prompts.py 
2. python generate_solutions.py

Reminders:
1. Create a .env file with GOOGLE_API_KEY=YOUR_API_KEY_HERE

## TODOs
- [ ] Update README with instructions on setup and what packages are needed (if wrong)
- [ ] Create a script called generate_json to convert all images into json files
- [ ] Modify generate_solutions to have an option to use json file vs images
- [ ] Expand generate_prompts and generate_solutions for datasets other than MathVista