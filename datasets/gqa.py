import itertools
import time
from logging import Logger

from tqdm import tqdm

from configs.config import Config
from src.captions_attributes import AttributeCaptionBuilder
from src.captions_reasoning import CountCaptionBuilder, ReasoningCaptionBuilder
from src.captions_relations import RelationCaptionBuilder
from src.sg_analysis import *
from src.templates import *


def generate_captions(
    attr_cpt_builder: AttributeCaptionBuilder,
    rel_cpt_builder: RelationCaptionBuilder,
    count_cpt_builder: CountCaptionBuilder,
    reasoning_cpt_builder: ReasoningCaptionBuilder,
    analysis: dict,
    caption_counter: int,
    cfg: Config,
    global_counter: int,
    logger: Logger,
) -> dict:
    """generates captions for one image/scene graph

    Args:
        attr_cpt_builder (AttributeCaptionBuilder):
        rel_cpt_builder (RelationCaptionBuilder):
        count_cpt_builder (CountCaptionBuilder):
        reasoning_cpt_builder (ReasoningCaptionBuilder):
        analysis (dict):
        caption_counter (int):
        cfg (Config):
        global_counter (int):
        logger (Logger):

    Returns:
        dict: contains scene graph for one particular image/scene graph
    """
    start_time = time.time()
    attr_captions = attr_cpt_builder.build_attr_captions(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for attr_captions:          {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    subgraph_attr_captions = attr_cpt_builder.build_subgraph_attr_captions(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for subgraph_attr_captions: {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    rel_captions = rel_cpt_builder.build_relation_captions(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for rel_captions:           {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    attr_rel_captions = rel_cpt_builder.build_attribute_relation_captions(
        analysis, caption_counter, cfg.get("rel_attr_patience"), global_counter
    )
    logger.info(
        f"Execution time for attr_rel_captions:      {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    obj_quantities = count_cpt_builder.build_obj_count_captions(
        analysis, caption_counter, cfg.get("obj_quant_threshold"), global_counter
    )
    logger.info(
        f"Execution time for obj_quantities:         {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    count_compare = count_cpt_builder.build_obj_compare_count_captions(
        analysis, caption_counter, cfg.get("obj_quant_error_margin"), global_counter
    )
    logger.info(
        f"Execution time for count_compare:          {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    verify_count = count_cpt_builder.build_verify_count_subgraph(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for verify_count:           {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    verify_attr_count = count_cpt_builder.build_verify_count_attr(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for verify_attr_count:      {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    and_logic_attr = reasoning_cpt_builder.build_and_logic_attr(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for and_logic_attr:         {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    and_logic_rel = reasoning_cpt_builder.build_and_logic_rel(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for and_logic_rel:          {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    xor_logic_attr = reasoning_cpt_builder.build_xor_logic_attr(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for xor_logic_attr:         {(time.time() - start_time):.2f}s"
    )
    start_time = time.time()
    xor_logic_rel = reasoning_cpt_builder.build_xor_logic_rel(
        analysis, caption_counter, global_counter
    )
    logger.info(
        f"Execution time for xor_logic_rel:          {(time.time() - start_time):.2f}s"
    )

    cpts_set = (
        subgraph_attr_captions
        | attr_captions
        | obj_quantities
        | rel_captions
        | attr_rel_captions
        | count_compare
        | verify_count
        | verify_attr_count
        | and_logic_attr
        | and_logic_rel
        | xor_logic_attr
        | xor_logic_rel
    )

    total_len = (
        len(subgraph_attr_captions)
        + len(attr_captions)
        + len(rel_captions)
        + len(attr_rel_captions)
        + len(obj_quantities)
        + len(count_compare)
        + len(verify_count)
        + len(verify_attr_count)
        + len(and_logic_attr)
        + len(and_logic_rel)
        + len(xor_logic_attr)
        + len(xor_logic_rel)
    )

    assert (
        len(cpts_set) == total_len
    ), "merging of captions dicts failed, possibly due to a caption counter error"
    return cpts_set


def run_gqa(
    gqa_sg: dict,
    cfg: Config,
    global_counter: int,
    logger: Logger,
    filter_noisy: bool,
    relaxed_mode: bool,
    run_subset: bool = False,
) -> dict:
    """loops over all instances (images or scene graphs) and triggers creation of captions

    Args:
        gqa_sg (dict): contains the scene graphs
        cfg (Config):
        global_counter (int):
        logger (Logger):
        run_subset (bool, optional): for debugging purposes. Defaults to False.

    Returns:
        dict: contains all captions, i.e. the finished datasets
    """
    attr_cpt_builder = AttributeCaptionBuilder(logger, filter_noisy, relaxed_mode)
    rel_cpt_builder = RelationCaptionBuilder(logger, filter_noisy, relaxed_mode)
    count_cpt_builder = CountCaptionBuilder(logger, filter_noisy, relaxed_mode)
    reasoning_cpt_builder = ReasoningCaptionBuilder(logger, filter_noisy, relaxed_mode)
    caption_counter = [0]
    dataset = {}

    if run_subset:
        gqa_sg = dict(itertools.islice(gqa_sg.items(), 1000))

    for i, img_id in enumerate(tqdm(gqa_sg)):
        scene_graph = gqa_sg[img_id]
        analysis = analyze_objects(scene_graph)
        cpts_set = generate_captions(
            attr_cpt_builder,
            rel_cpt_builder,
            count_cpt_builder,
            reasoning_cpt_builder,
            analysis,
            caption_counter,
            cfg,
            global_counter,
            logger,
        )
        dataset[img_id] = {"captions": cpts_set}
    return dataset
