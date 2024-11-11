from nerif.core import Nerification, NerificationInt, NerificationString

nerification = Nerification(model="text-embedding-3-large")

print(nerification.simple_fit("yes, it is"))
# result: None
print(nerification.force_fit("yes, it is"))
# result: True
print(nerification.simple_fit("true"))
# result: True
print(nerification.force_fit("true"))
# result: True

nerification_int = NerificationInt(model="text-embedding-3-large", possible_values=[1, 233, 343])


print(nerification_int.simple_fit(1))
# result: 1
print(nerification_int.force_fit(1))
# result: 1
print(nerification_int.simple_fit(233))
# result: 233
print(nerification_int.force_fit("The value is 233"))
# result: 233
print(nerification_int.simple_fit(343))
# result: 343
print(nerification_int.force_fit("The value is 343"))
# result: 343

nerification_string = NerificationString(model="text-embedding-3-large", possible_values=["YES", "NO"])

print(nerification_string.simple_fit("yes"))
# result: YES
print(nerification_string.force_fit("Well, I guess you are right"))
# result: YES
print(nerification_string.simple_fit("no"))
# result: NO
print(nerification_string.force_fit("Oh, I don't think so"))
# result: NO
