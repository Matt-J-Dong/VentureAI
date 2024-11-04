# VentureAI
VentureAI

## Cleaning:
All instances of `&quot;` have been replaced with `\"` to since we wouldn't want the model to output special formatting like that.
All information specific to images inside double brackets `{{}}` has been removed, but the captions for the images currently are kept because they still contain useful information.
There are two different variations of `[[]]`. The ones that contain the substring `File:` within them are file links, and those have been removed. Otherwise, they are links to other pages in WikiVoyage. For those, we just removed the brackets and kept the text.