# Goal
- Create a README for the audio submodule

# Guidelines
## Structure
- explain what the purpose of the audio module does in the greater context of the overall program
- Brief explanantion as to why we overlap the audio in the first place
- Explain how it works
- Explain briefly how it gets configured and init

Example
### Key Elements:
- Showcase how the audio overlap scheme works, using an interactive example that shows how we grab one window from the audio, then put a second window on it, overlapping the previous window, interactively update the plot to animate the progression of the windows

- Explains how edge discontinuities introduce aliasing and high frequency energy that's not actually there we motivate ourselves to have an overlap
- Example of no overlap vis overlap
- Animated example of the the audio window sliding over the audio - this is already in the benchmark file but it would be best here in theis jupyter notebook

- **4-Row Overlap Visualization:** 
  - Row 1: Original audio signal
  - Row 2: Orange window (first chunk)
  - Row 3: Blue window (second chunk) 
  - Row 4: Combined result showing replacement strategy
- **Edge Effect Mitigation:** How overlap reduces artifacts
- **Window/Overlap Relationships:** Interactive parameter exploration


## Style
- Simple concise language 
- The goal is to simplify what we're doing and explain why we are doing
- "We want to reduce __ so we do __"
- "Now that we've done __, this sets us up nicely for the next time we __", 


## Focus:
Visual demonstration of windowing strategy without diving into other modules