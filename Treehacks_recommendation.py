from googlesearch import search
import openai


def googleSearchURL(list_words):
    result_URL = []
    for i in list_words:
        query = i.split()
        searchquery = query[0]
        for j in query[1:]:
            searchquery = searchquery + " AND " + j
        for result in search(searchquery, num_results=2):
            result_URL.append(result)
    return result_URL[1:]


def textsummary(input_string):
    input_prompt = "Explain the content below in simpler terms:\n\n" + input_string
    openai.api_key = "sk-UQuGm7CmM87xbJWSKAsZ4QqYkeBTz63JOcR00ZJk"
    response = openai.Completion.create(
      engine="davinci-instruct-beta",
      prompt=input_prompt,
      temperature=1,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0]["text"].strip().replace("\n", "")
#
# list1 = ["Coginitive Science Stanford","The Last Supper"]
# test_google_links = googleSearchURL(list1)
# test_OpenAI = textsummary(list1)
#
# print(test_google_links)
# print(test_OpenAI)
