# Visualizer README Outline

## Instructions
- Analyze the plotter code with Opus
- Explain how this works in the greater context of the pipeline
- Explain why we use the shader instead of a typical plotting libraries (way too many points for python)
- Explain how its init and configured
- Explain what happens at run time
- Explain how the overlap is handled externally, and we just fill a circular buffer of frames
- Diagram (I already have this) place holder, when we get here, ask me to provide the diagram
- Talk about the CPU-GPU transfers happening during init, how we do as much as possible up front, and then during runtime, transfer the bare minimum
- Discuss normalizing the data for the shader colormap, how we keep track of a max value so all individual plots a relative to each other
- Brief gamma correction to adjust - eventually will set to a constant once we lock in the gamma we like/is the best / most accurate

## Interaction Protocol
- When the human gives you text to condense: preserve their phrasing where possible, cut repetition and filler, keep the best version of repeated concepts in the position that makes logical sense
- When the human talks through an idea verbally (voice transcripts): extract the core points, discard false starts and filler words, present back as structured content for their review
- When asked for an outline: give structure only, no prose, no descriptions of what each section "will cover"
- When asked to draft: write the actual content, not a description of what you'd write
- Flag factual uncertainties rather than guessing — say "I'd need to check the code for X"

### Philosophy:
*"If you understand vectors and pattern matching, wavelets are just a natural extension"*
