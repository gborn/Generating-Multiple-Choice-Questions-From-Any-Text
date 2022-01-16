
from src.answerkey import AnswerKey
from src.model import Model, unmasker
import streamlit as st

PAGE_CONFIG = {"page_title":"MCQ-App by Glad Nayak","page_icon":":white_check_mark:"}
st.set_page_config(**PAGE_CONFIG)

def render_input():
    """
    Renders text area for input, and button
    """
    # source of default text: https://www.fresherslive.com/online-test/reading-comprehension-test-questions-and-answers
    text = """The Dust Bowl, considered one of the greatest man-made ecological disasters, was a period of severe dust storms that lasted nearly a decade, starting 1931, and engulfed large parts of the US. The dust storms originated in the Great Plains-from states like Texas, Oklahoma, New Mexico, Colorado and Kansas. They were so severe that they choked everything and blocked out the sun for days. Sometimes, the storms travelled thousands of kilometres and blotted out monuments such as the Statue of Liberty. Citizens developed “dust pneumonia” and experienced chest pain and difficulty in breathing. The storms damaged the soil in around 100 million acres of land, leading to the greatest short-time migration in the American history, with approximately 3.5 million people abandoning their farms and fields.

    Dust storms are an annual weather pattern in the northern region of India comprising Delhi, Haryana, Punjab, Uttar Pradesh and Rajasthan and Punjab, as also in the Sindh region of Pakistan. But, they are normally low in intensity and accompanied by rains. In fact, people welcome dust storms as they bring down temperatures and herald the arrival of the monsoons. But, the dust storms that have hit India since February this year have been quantitatively and qualitatively different from those in the past. They are high-powered storms travelling long distances and destroying properties and agricultural fields. Since February, they have affected as many as 16 states and killed more than 500 people. Cities like Delhi were choked in dust for days, with air quality level reaching the “severe” category on most days.

    The Dust Bowl areas of the Great Plains are largely arid and semi-arid and prone to extended periods of drought. The US federal government encouraged settlement and development of large-scale agriculture by giving large parcels of grasslands to settlers. Waves of European settlers arrived at the beginning of the 20th century and converted grasslands into agricultural fields. At the same time, technological improvements allowed rapid mechanization of farm equipment, especially tractors and combined harvesters, which made it possible to operate larger parcels of land.

    For the next two decades, agricultural land grew manifold and farmers undertook extensive deep ploughing of the topsoil with the help of tractors to plant crops like wheat. This displaced the native, deep-rooted grasses that trapped soil and moisture even during dry periods and high winds. Then, the drought struck. Successive waves of drought, which started in 1930 and ended in 1939, turned the Great Plains into bone-dry land. As the soil was already loose due to extensive ploughing, high winds turned them to dust and blew them away in huge clouds. Does this sound familiar? The dust storm regions of India and Pakistan too are largely arid and semi-arid. But they are at a lower altitude and hence less windy compared to the Great Plains. Over the last 50 years, chemical- and water-intensive agriculture has replaced the traditional low-input agriculture. Canal irrigation has been overtaken by the groundwater irrigation. In addition, mechanized agriculture has led to deeper ploughing, loosening more and more topsoil. The result has been devastating for the soil and groundwater. In most of these areas, the soil has been depleted and groundwater levels have fallen precipitously. On top of the man-made ecological destruction, the natural climatic cycle along with climate change is affecting the weather pattern of this region.

    First, this area too is prone to prolonged drought. In fact, large parts of Haryana, Punjab, Delhi and western UP have experienced mildly dry to extremely dry conditions in the last six years. The Standardized Precipitation Index (SPI), which specifies the level of dryness or excess rains in an area, of large parts of Haryana, Punjab and Delhi has been negative since 2012. Rajasthan, on the other hand shows a positive SPI or excess rainfall. Second, this area is experiencing increasing temperatures. In fact, there seems to be a strong correlation between the dust storms and the rapid increase in temperature. Maximum temperatures across northern and western India have been far higher than normal since April this year. Last, climate change is affecting the pattern of Western Disturbances (WDs), leading to stronger winds and stronger storms. WDs are storms originating in the Mediterranean region that bring winter rain to northwestern India. But because of the warming of the Arctic and the Tibetan Plateau, indications are that the WDs are becoming unseasonal, frequent and stronger.

    The Dust Bowl led the US government to initiate a large-scale land-management and soil-conservation programme. Large-scale shelterbelt plantations, contour ploughing, conservation agriculture and establishment of conservation areas to keep millions of acres as grassland, helped halt wind erosion and dust storms. It is time India too recognizes its own Dust Bowl and initiates a large-scale ecological restoration programme to halt it. Else, we will see more intense dust storms, and a choked Delhi would be a permanent feature.
    """
    st.sidebar.subheader('Enter Text:')
    text = st.sidebar.text_area('', text.strip(), height = 275)

    ngram_range = st.sidebar.slider('answer ngram range:', value=[1, 2], min_value=1, max_value=3, step=1)
    num_questions = st.sidebar.slider("number of questions:", value=10, min_value=10, max_value=20, step=1)
    question_type_str = st.sidebar.radio('question type:', ('declarative (fill in the blanks)', 'imperative'))
    question_type = question_type_str == 'declarative (fill in the blanks)'

    button = st.sidebar.button('Generate')

    if button:
        return (text, ngram_range, num_questions, question_type)

def main():
    # Render input text area
    inputs = render_input()

    if not inputs:
        st.title('Generate Multiple Choice Questions(MCQs) from Text Automatically')
        st.subheader('Enter Text, select how long a single answer should be(ngram_range), and number of questions to get started.')

    else:
        with st.spinner('Loading questions and distractors using BERT model'):
            st.subheader("")
            st.title("")
            text, ngram_range, num_questions, question_type = inputs

            # Load model
            answerkeys = AnswerKey(text)
            keyword_to_sentence = answerkeys.get_answers(ngram_range, num_questions)

            model = Model()
            quizzes = model.get_questions(keyword_to_sentence, unmasker, k=num_questions, declarative=question_type)

            st.subheader('Questions')
            for id, quiz in enumerate(quizzes):
                question, options, answer = quiz
                st.write(question)

                for option in options[:3]:
                    st.checkbox(option, key=id)

                ans_button = st.checkbox(answer, key=id, value=True)

            st.balloons()
            st.button('Save')


if __name__ == '__main__':
    main()
        