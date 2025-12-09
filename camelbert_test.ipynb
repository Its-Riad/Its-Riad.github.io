from transformers import pipeline

print("Loading CAMeL-BERT...")
camelbert = pipeline("sentiment-analysis", 
    model="CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment")

# Test 1: CLEARLY NEGATIVE (crisis, disaster, collapse)
negative_text = """
أزمة اقتصادية خانقة تضرب مصر مع ارتفاع معدل التضخم إلى مستويات 
كارثية تصل إلى 35 بالمئة. الوضع يثير قلقا بالغا ومخاوف كبيرة 
من انهيار اقتصادي وشيك. الخبراء يحذرون من كارثة وشيكة.
"""

# Test 2: CLEARLY POSITIVE (growth, success, improvement)
positive_text = """
الاقتصاد المصري يحقق نموا قويا ومبهرا مع انخفاض ملحوظ في معدل 
التضخم إلى أدنى مستوياته. تحسن كبير في جميع المؤشرات الاقتصادية 
يبشر بمستقبل مشرق ومزدهر للبلاد.
"""

# Test 3: NEUTRAL (just facts)
neutral_text = """
أعلن البنك المركزي المصري عن عقد اجتماع اليوم لمناقشة السياسة 
النقدية. البنك سيصدر بيانا بعد الاجتماع.
"""

print("\n" + "="*60)
print("TEST 1: CLEARLY NEGATIVE TEXT")
print("="*60)
result1 = camelbert(negative_text)[0]
print(f"Text: {negative_text[:80]}...")
print(f"Result: {result1['label']} (confidence: {result1['score']:.2f})")

print("\n" + "="*60)
print("TEST 2: CLEARLY POSITIVE TEXT")
print("="*60)
result2 = camelbert(positive_text)[0]
print(f"Text: {positive_text[:80]}...")
print(f"Result: {result2['label']} (confidence: {result2['score']:.2f})")

print("\n" + "="*60)
print("TEST 3: NEUTRAL TEXT")
print("="*60)
result3 = camelbert(neutral_text)[0]
print(f"Text: {neutral_text[:80]}...")
print(f"Result: {result3['label']} (confidence: {result3['score']:.2f})")
