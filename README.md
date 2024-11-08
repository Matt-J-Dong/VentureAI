# VentureAI

![VentureAI Logo](./VentureAI.png)

We introduce Venture AI, your automated travel agent. The goal of Venture AI is to address the complexity of modern travel planning by creating a single application that provides personalized travel recommendations tailored to each user’s preferences and budget. Today’s large language models are very versatile, but they lack the specific customization needed to deliver precise suggestions for unique user needs. By utilizing LLMs and integrating them with real-time APIs from various travel-related sites, Venture AI will act as a comprehensive travel planner that can cater to a wide range of travel preferences.

This project addresses a common problem with travel planning by using new LLM techniques to simplify and centralize the planning process. Without LLMs, users must manually navigate different websites to organize flights and local activities, often making travel planning inefficient and demotivating. LLMs offer an advantage in this space due to their ability to handle and respond to natural language queries. They can process real-time user feedback to refine suggestions in an adaptable and user-friendly way.

Through these capabilities, Venture AI aims to transform how travelers organize their adventures, making the process more efficient, personalized, and enjoyable.

## Cleaning:
* All instances of `&quot;` have been replaced with `\"` to since we wouldn't want the model to output special formatting like that.
* All information specific to images inside double brackets `{{}}` has been removed, but the captions for the images currently are kept because they still contain useful information.
* There are two different variations of `[[]]`. The ones that contain the substring `File:` within them are file links, and those have been removed. Otherwise, they are links to other pages in WikiVoyage. For those, we just removed the brackets and kept the text.

* Working on how to format this information: `*  name=The North Brabant Museum | alt=NoordBrabants Museum | url=http://www.hetnoordbrabantsmuseum.nl/english | email=\n| wikidata= Q12013217\n| address= | lat=51.68658 | long=5.30469 | directions=\n| phone= | tollfree= | fax=\n| hours= | price=\n| lastedit=2016-01-25\n| content=It houses a collection of art and historical artifacts, from pre-roman times to the 20th century. Special exibitions are a must to see, the 'Hyeronimus Bosch Exhibition' with his original work from museums all over the world.\n\n\n\n`. Is the star part of this section or the next section? What do all the new lines do? How can we take this information so the model can potentially use it?