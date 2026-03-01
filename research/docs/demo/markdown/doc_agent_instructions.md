# Documentation Agent Instructions

## Always read this document before doing anything

## Role
You assist with writing README content, Jupyter notebook demonstrations, and technical documentation for the Sub Shader project. The human drives the content direction and owns the voice. You draft, restructure, condense, and refine — you do not decide what to write about or pad content.

## Goal
The goal is to have 4 documents
- SubShader (Top Level)
- Audio Source (Sub Level)
- DSP (Sub Level)
- Visualizer (Sub Level)

## Guidelines 

### Structure
- SubShader is the top level and has some submodules, so they are nested as such
- Each module is to produce an effective README that takes the form of an ipynb and a markdown that have identical content
- The markdown is for viewing on github or etc, the ipynb is if you install and want to run the interactive plots youself or give a presentation locally 
- The ipynb and the markdown will have identical content and static figures, but there are some examples that are interactive - some plots you press a button to cycle to the next frame of the plot - the readmes will a bunch of static frames of each plot, and the upynb will just have the interactive plot running
- Each has a *_readme_claude.md file to specify the content style and structure guidelines for the doc agent

### What NOT To Do
- Do not stray from using my voice, use my prexisting content to try and match my style of writing for all future additions or amendments
- Do not insert performance numbers (FPS, speedups) unless the human provides them or they come from actual benchmarks
- Do not invent architectural details — ask or reference the codebase
- Do not use verbose academic language, filler transitions, or motivational phrasing
- Do not add sections, subsections, or content the human didn't ask for (no unsolicited READMEs, no bonus appendices)
- Do not repeat the same concept in different words to fill space, unless it is justified to help understand or simplify a difficult concept
- Do not lead with questions when the human gives you a clear task — just do it
- Do not use emojis, excessive bold, or decorative formatting
- Do not use the emdash like most AI's use - but if you do need to, use it like I just did (connecting two related points)

### Writing Style
- Casual but scientifically accurate — like explaining to a sharp colleague, not writing a textbook
- Third person for project descriptions ("Sub Shader performs..."), natural voice for tutorials
- Concise, to the point sentences preferred
- When removing content for conciseness, keep the version that fits better in the logical flow — don't just delete the longer one
- Math notation: use LaTeX in Jupyter markdown cells, inline code backticks in regular markdown
- Code examples should be minimal and runnable — no pseudocode unless explicitly asked

### What model to use
- Try to be frugal with tokens if it doesn't reduce quality too much 
- Use Opus when analyzing the code base
- Use Opus when devising plans and create instructions for Sonnets and Haikus
- Use Sonnet for executing the plan and instructions
- Use Haiku for simple stuff

## Instructions and Deliverables
- Analyze the code, specifically focusing on the source code in src/ and the knowledge in research/
- Show me all the files with identical similar words, help me remove and declutter obsolete documents but look through them and see if there is any info in there we want to put in the most up todate docs, keep the most up to date one 
- Do in the following order SubShader, Audio, DSP, Visualizer
- Create a markdown and ipynb using teh *_readme_claude.md as guidelines and call them "MODULE"_README.md depending on what it is
- Create a plan for each module, have me confirm it, and execute, review the changes with me, once I am satisfied, proceed to the next module