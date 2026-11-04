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


def _rigify(prefix):
    p = {}
    def add(n, par=None):
        p[prefix + n] = (prefix + par) if par else None
    add("spine"); add("pelvis.L", "spine"); add("pelvis.R", "spine")
    prev = "spine"
    for i in range(1, 7):
        n = f"spine.{i:03d}"; add(n, prev); prev = n
    for s in "LR":
        add(f"shoulder.{s}", "spine.003"); add(f"upper_arm.{s}", f"shoulder.{s}")
        add(f"upper_arm.{s}.001", f"upper_arm.{s}"); add(f"forearm.{s}", f"upper_arm.{s}.001")
        add(f"hand.{s}", f"forearm.{s}"); add(f"thigh.{s}", "spine"); add(f"shin.{s}", f"thigh.{s}")
        add(f"foot.{s}", f"shin.{s}"); add(f"toe.{s}", f"foot.{s}")
        add(f"thigh_fk.{s}", "spine"); add(f"upper_arm_ik.{s}", "spine")
    return p


@pytest.mark.parametrize("prefix", ["DEF-", "ORG-", ""])
def test_rigify(prefix):
    m = map_rig(_rigify(prefix))
    assert m.ok, m.missing()
    assert m.hips == prefix + "spine" and m.head == prefix + "spine.006"
    assert m.spine == [prefix + f"spine.00{i}" for i in (1, 2, 3)]
    assert m.neck == [prefix + "spine.004", prefix + "spine.005"]
    assert m.limbs["L"]["upperarm"] == prefix + "upper_arm.L"
    assert m.limbs["R"]["shin"] == prefix + "shin.R" and m.limbs["R"]["toe"] == prefix + "toe.R"


def test_assign_roles_by_hand():
    from openrig.core.mapping import RigMap
    m = RigMap()
    m.assign("hips", ["pelvis"]); m.assign("spine", ["s1", "s2"]); m.assign("head", ["head"])
    m.assign("forearm.L", ["lower_arm_l"])
    assert m.hips == "pelvis" and m.spine == ["s1", "s2"] and m.limbs["L"]["forearm"] == "lower_arm_l"
    m.assign("toe.L", ["ball"]); m.assign("toe.L", [])
    assert "toe" not in m.limbs["L"]
    with pytest.raises(ValueError):
        m.assign("wing.L", ["x"])
    labels = [r[0] for r in m.rows()]
    assert "Hips" in labels and "Forearm L" in labels


def test_assign_validates_input():
    from openrig.core.mapping import RigMap
    m = RigMap()
    m.assign("hips", ["pelvis"])
    with pytest.raises(ValueError):
        m.assign("hand.L", ["a", "b"])            # one bone only
    with pytest.raises(ValueError):
        m.assign("spine", ["pelvis", "s1"])        # pelvis already hips
    with pytest.raises(ValueError):
        m.assign("spine", ["s1", "s1"])
    m.assign("hips", ["pelvis"])                   # re-assigning the same role is fine


def test_partial_rigify_falls_back_to_name_matching():
    p = {"spine": None, "spine.006": "spine", "Hips": "spine"}
    m = map_rig(p)
    assert m.hips != "spine"       # generic path ran instead of returning a half-empty Rigify map
