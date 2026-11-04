import pytest
from openrig.core.mapping import map_rig, classify

def parents_from(names_parents):
    return dict(names_parents)

def test_classify_mixamo():
    assert classify("mixamorig:LeftArm") == ("upperarm", "L")
    assert classify("mixamorig:LeftForeArm") == ("forearm", "L")
    assert classify("mixamorig:LeftUpLeg") == ("thigh", "L")
    assert classify("mixamorig:LeftLeg") == ("shin", "L")
    assert classify("mixamorig:RightToeBase") == ("toe", "R")
    assert classify("mixamorig:LeftShoulder") == ("clavicle", "L")
    assert classify("mixamorig:LeftHandIndex1") is None
    assert classify("mixamorig:HeadTop_End") is None
    assert classify("mixamorig:LeftToe_End") is None

def test_classify_ue5():
    assert classify("upperarm_l") == ("upperarm", "L")
    assert classify("lowerarm_r") == ("forearm", "R")
    assert classify("calf_l") == ("shin", "L")
    assert classify("ball_r") == ("toe", "R")
    assert classify("thigh_twist_01_l") is None
    assert classify("ik_foot_l") is None
    assert classify("spine_03") == ("spine", "C")
    assert classify("pelvis") == ("hips", "C")

def test_classify_other_rigs():
    assert classify("Bip01 L Thigh") == ("thigh", "L")
    assert classify("Bip01 L Calf") == ("shin", "L")
    assert classify("Bip01 R UpperArm") == ("upperarm", "R")
    assert classify("Bip01 Pelvis") == ("hips", "C")
    assert classify("J_Bi_L_UpperLeg") == ("thigh", "L")
    assert classify("J_Bi_L_LowerLeg") == ("shin", "L")
    assert classify("J_Bi_L_Shoulder") == ("clavicle", "L")
    assert classify("J_Bi_L_ToeBase") == ("toe", "L")
    assert classify("CC_Base_L_Upperarm") == ("upperarm", "L")
    assert classify("CC_Base_R_Calf") == ("shin", "R")
    assert classify("lShldrBend") == ("upperarm", "L")
    assert classify("rThighBend") == ("thigh", "R")
    assert classify("lForearmBend") == ("forearm", "L")
    assert classify("thigh.L") == ("thigh", "L")
    assert classify("upper_arm.R") == ("upperarm", "R")

def test_full_map_ue5():
    p = {"root": None, "pelvis": "root", "spine_01": "pelvis", "spine_02": "spine_01",
         "spine_03": "spine_02", "neck_01": "spine_03", "head": "neck_01",
         "clavicle_l": "spine_03", "upperarm_l": "clavicle_l", "lowerarm_l": "upperarm_l",
         "hand_l": "lowerarm_l", "clavicle_r": "spine_03", "upperarm_r": "clavicle_r",
         "lowerarm_r": "upperarm_r", "hand_r": "lowerarm_r", "thigh_l": "pelvis",
         "calf_l": "thigh_l", "foot_l": "calf_l", "ball_l": "foot_l", "thigh_r": "pelvis",
         "calf_r": "thigh_r", "foot_r": "calf_r", "ball_r": "foot_r",
         "thigh_twist_01_l": "thigh_l", "ik_foot_l": "root"}
    m = map_rig(p)
    assert m.ok, m.missing()
    assert m.hips == "pelvis" and m.spine == ["spine_01", "spine_02", "spine_03"]
    assert m.limbs["L"]["shin"] == "calf_l" and m.limbs["R"]["toe"] == "ball_r"

def test_charactercreator_neck_twist_fallback():
    p = {"CC_Base_Pelvis": None, "CC_Base_Waist": "CC_Base_Pelvis",
         "CC_Base_NeckTwist01": "CC_Base_Waist", "CC_Base_Head": "CC_Base_NeckTwist01"}
    m = map_rig(p)
    assert m.neck == ["CC_Base_NeckTwist01"]

def test_missing_reported():
    m = map_rig({"Hips": None, "Spine": "Hips"})
    assert not m.ok and "head" in m.missing()

def test_roundtrip_dict():
    from openrig.core.mapping import RigMap
    from rigs import make
    _, m, _, _ = make()
    assert RigMap.from_dict(m.to_dict()).to_dict() == m.to_dict()
