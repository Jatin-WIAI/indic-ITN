# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
from inverse_text_normalization.en.data_loader_utils import get_abs_path
from inverse_text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.en.utils import num_to_word
import json

data_path = 'data/'

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

def get_graph(keys,value):
    graph_list = []
    for key in keys:
        # print("graph : ", key, value)
        temp_graph = pynini.cross(key, value)
        graph_list.append(temp_graph)
    return pynini.union(*graph_list)

def get_delete_graph(keys):
    graph_list = []
    for key in keys:
        temp_graph = pynutil.delete(key)
        graph_list.append(temp_graph)
    return pynini.union(*graph_list)

class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted. 
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        # integer, negative

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        order_dict = open(get_abs_path(data_path + "numbers/order_dict.json"), "r")
        order_dict = json.load(order_dict)

        graph_hundred = pynini.cross("hundred", "")

        # handling double digit hundreds like उन्निस सौ + digit/thousand/lakh/crore etc
        graph_hundred_component_prefix_tens = pynini.union(graph_digit + delete_space + get_delete_graph(order_dict["hundred"]["keys"]) + delete_space,)
                                                           # pynutil.insert("55"))
        graph_hundred_component_prefix_tens += pynini.union(graph_ties,
                                                            pynutil.insert("0") + (graph_digit | pynutil.insert("0")))
        

        graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("0"))
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        graph_hundred_component_in = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert(""))
        graph_hundred_component_in += delete_space
        graph_hundred_component_in += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        # graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
        #     pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        # )
        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("thousand"),
            pynutil.insert("000", weight=0.1),
        )

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("million"),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("billion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("trillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("quadrillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("quintillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("sextillion"),
            pynutil.insert("000", weight=0.1),
        )

        graph = pynini.union(
            graph_sextillion
            + delete_space
            + graph_quintillion
            + delete_space
            + graph_quadrillion
            + delete_space
            + graph_trillion
            + delete_space
            + graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        graph_thousands_component = pynini.union(
            graph_hundred_component_in + delete_space + pynutil.delete("thousand"),
            pynutil.insert("00", weight=0.1),
        )

        graph_lakhs_component = pynini.union(
            graph_hundred_component_in + delete_space + pynutil.delete("lakh"),
            pynutil.insert("00", weight=0.1)
        )
        #
        graph_crores_component = pynini.union(
            graph_hundred_component_in + delete_space + pynutil.delete("crore"),
            pynutil.insert("000", weight=0.1)
        )

        # some special words like dedh, etc which donot follow a pattern.
        special_words_dict = open(get_abs_path(data_path + "numbers/special_words_dict.json"), "r")
        special_words_dict = json.load(special_words_dict)
        special_words_graphs = []
        for k,v in special_words_dict.items():
            special_words_graphs.append(get_graph(v,k))
        special_words_graph = pynini.union(*special_words_graphs)

        #fraction word. 
        fraction_words_dict = open(get_abs_path(data_path + "numbers/fraction_words_dict.json"), "r")
        fraction_words_dict = json.load(fraction_words_dict)
        fraction_word_graph= get_graph(fraction_words_dict["FRACX.5"],"FRACX.5")


        #higher order fractions
        higher_order_fraction_graphs_1 = get_graph(fraction_words_dict["FRAC1.5"],"0")+delete_space+(get_graph(order_dict["hundred"]["keys"],"150") | get_graph(order_dict["thousand"]["keys"],"1500") | get_graph(order_dict["lakh"]["keys"],"150000") | get_graph(order_dict["crore"]["keys"],"15000000"))
        higher_order_fraction_graphs_2 = get_graph(fraction_words_dict["FRAC2.5"],"0")+delete_space+(get_graph(order_dict["hundred"]["keys"],"250") | get_graph(order_dict["thousand"]["keys"],"2500") | get_graph(order_dict["lakh"]["keys"],"250000") | get_graph(order_dict["crore"]["keys"],"25000000"))


        #
        graph_in = pynini.union(
            graph_crores_component
            + delete_space
            + graph_lakhs_component
            + delete_space
            + graph_thousands_component
            + delete_space
            + graph_hundred_component,
            graph_zero,
            special_words_graph,
            fraction_word_graph,
            higher_order_fraction_graphs_1,
            higher_order_fraction_graphs_2
        )

        graph = pynini.union(graph, graph_in)
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT), "0"
        )

        labels_exception = [num_to_word(x) for x in range(0, 13)]
        graph_exception = pynini.union(*labels_exception)

        graph = pynini.cdrewrite(pynutil.delete("and"), NEMO_SPACE, NEMO_SPACE, NEMO_SIGMA) @ graph

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pynini
from pynini.lib import pynutil, utf8
import json
from inverse_text_normalization.en.data_loader_utils import get_abs_path
from inverse_text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.en.utils import num_to_word
# from inverse_text_normalization.lang_params import LANG
# data_path = f'data/{LANG}_data/'
data_path = 'data/'

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

def get_graph(keys,value):
    graph_list = []
    for key in keys:
        temp_graph = pynini.cross(key, value)
        graph_list.append(temp_graph)
    return pynini.union(*graph_list)

def get_delete_graph(keys):
    graph_list = []
    for key in keys:
        temp_graph = pynutil.delete(key)
        graph_list.append(temp_graph)
    return pynini.union(*graph_list)

class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted. 
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        # integer, negative

        NEMO_CHAR = utf8.VALID_UTF8_CHAR
        NEMO_SIGMA = pynini.closure(NEMO_CHAR)
        NEMO_SPACE = " "
        NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", u"\u00A0").optimize()
        NEMO_NOT_SPACE = pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE).optimize()
        # NEMO_NON_BREAKING_SPACE = u"\u00A0"

        hindi_digit_file = get_abs_path(data_path + 'numbers/digit.tsv')
        with open(hindi_digit_file) as f:
            digits = f.readlines()
        hindi_digits = ''.join([line.split()[-1] for line in digits])
        # hindi_digits_with_zero = "0" + hindi_digits
        # print(f'hindi digits is {hindi_digits}')
        # HINDI_DIGIT = pynini.union(*hindi_digits).optimize()
        # HINDI_DIGIT_WITH_ZERO = pynini.union(*hindi_digits_with_zero).optimize()

        graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/zero.tsv"))
        graph_tens = pynini.string_file(get_abs_path(data_path + "numbers/tens.tsv"))
        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        # | pynini.string_map(("एक","1")) | pynini.string_map(("दो","2"))


        order_dict = open(get_abs_path(data_path + "numbers/order_dict.json"), "r")
        order_dict = json.load(order_dict)
        graph_hundred = get_graph(order_dict["hundred"]["keys"],"00")
        graph_crore =get_graph(order_dict["crore"]["keys"],"0000000")
        graph_lakh = get_graph(order_dict["lakh"]["keys"],"00000")
        graph_thousand  =get_graph(order_dict["thousand"]["keys"],"000")

        graph_hundred_component = pynini.union(graph_digit + delete_space + get_delete_graph(order_dict["hundred"]["keys"]) + delete_space,
                                               pynutil.insert("0"))
        graph_hundred_component += pynini.union(graph_tens, pynutil.insert("0") + (graph_digit | pynutil.insert("0")))

        # handling double digit hundreds like उन्निस सौ + digit/thousand/lakh/crore etc
        graph_hundred_component_prefix_tens = pynini.union(graph_tens + delete_space + get_delete_graph(order_dict["hundred"]["keys"]) + delete_space,)
                                                           # pynutil.insert("55"))
        graph_hundred_component_prefix_tens += pynini.union(graph_tens,
                                                            pynutil.insert("0") + (graph_digit | pynutil.insert("0")))

        # graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
        #         pynini.closure(HINDI_DIGIT_WITH_ZERO) + (HINDI_DIGIT_WITH_ZERO - "०") + pynini.closure(HINDI_DIGIT_WITH_ZERO)
        # )
        graph_hundred_component_non_hundred = pynini.union(graph_tens,
                                                           pynutil.insert("0") + (graph_digit | pynutil.insert("0")))

        graph_hundred_component = pynini.union(graph_hundred_component,
                                               graph_hundred_component_prefix_tens)

        graph_hundred_component_at_least_one_none_zero_digit = pynini.union(graph_hundred_component, graph_hundred_component_non_hundred)

        graph_solo_hundred_component = get_graph(order_dict["hundred"]["keys"],"100")
        graph_solo_thousand_component = get_graph(order_dict["thousand"]["keys"],"1000")
        graph_solo_lakh_component = get_graph(order_dict["lakh"]["keys"],"100000")
        graph_solo_crore_component = get_graph(order_dict["crore"]["keys"],"10000000")

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_thousands_component = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + get_delete_graph(order_dict["thousand"]["keys"]),
            pynutil.insert("00", weight=0.1),
        )

        graph_lakhs_component = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + get_delete_graph(order_dict["lakh"]["keys"]),
            pynutil.insert("00", weight=0.1)
        )

        graph_crores_component = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + get_delete_graph(order_dict["crore"]["keys"]),
            pynutil.insert("00", weight=0.1)
        )

        # some special words like dedh, etc which donot follow a pattern.
        special_words_dict = open(get_abs_path(data_path + "numbers/special_words_dict.json"), "r")
        special_words_dict = json.load(special_words_dict)
        special_words_graphs = []
        for k,v in special_words_dict.items():
            special_words_graphs.append(get_graph(v,k))
        special_words_graph = pynini.union(*special_words_graphs)

        #fraction word. 
        fraction_words_dict = open(get_abs_path(data_path + "numbers/fraction_words_dict.json"), "r")
        fraction_words_dict = json.load(fraction_words_dict)
        fraction_word_graph= get_graph(fraction_words_dict["FRACX.5"],"FRACX.5")

        #higher order fractions
        higher_order_fraction_graphs_1 = get_graph(fraction_words_dict["FRAC1.5"],"0")+delete_space+(get_graph(order_dict["hundred"]["keys"],"150") | get_graph(order_dict["thousand"]["keys"],"1500") | get_graph(order_dict["lakh"]["keys"],"150000") | get_graph(order_dict["crore"]["keys"],"15000000"))
        higher_order_fraction_graphs_2 = get_graph(fraction_words_dict["FRAC2.5"],"0")+delete_space+(get_graph(order_dict["hundred"]["keys"],"250") | get_graph(order_dict["thousand"]["keys"],"2500") | get_graph(order_dict["lakh"]["keys"],"250000") | get_graph(order_dict["crore"]["keys"],"25000000"))

        # fst = graph_thousands
        fst = pynini.union(
            graph_crores_component
            + delete_space
            + graph_lakhs_component
            + delete_space
            + graph_thousands_component
            + delete_space
            + graph_hundred_component,
            graph_zero,
            graph_solo_crore_component,
            graph_solo_lakh_component,
            graph_solo_thousand_component,
            graph_solo_hundred_component,
            special_words_graph,
            fraction_word_graph,
            higher_order_fraction_graphs_1,
            higher_order_fraction_graphs_2,
            graph_tens,
            graph_digit
        )

        fst_crore = fst+graph_crore # handles words like चार हज़ार करोड़
        fst_lakh = fst+graph_lakh # handles words like चार हज़ार लाख
        fst = pynini.union(fst, fst_crore, fst_lakh, graph_crore, graph_lakh, graph_thousand, graph_hundred)

        labels_exception = [num_to_word(x) for x in range(1, 3)]
        graph_exception = pynini.union(*labels_exception)

        self.graph_no_exception = fst

        self.graph = (pynini.project(fst, "input") - graph_exception.arcsort()) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()