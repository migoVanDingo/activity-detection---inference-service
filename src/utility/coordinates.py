def get_intersection_coordinates(rect1, rect2):
        """
        FUNCTION RENAMED FROM: _get_intersection()
        ORIGINAL FILE: framework3_roi_kb.py

        Get intersectin of two rectangles based on image coordinate
        system

        Parameters
        ----------
        rect1 : List[int]
            Coordinates of one bounding box
        rect2 : List[int]
            Coordinates of second bounding box.
        """
        
        # Intersection box coordinates
        wp_tl = rect1[0]
        hp_tl = rect1[1]
        wp_br = wp_tl + rect1[2]
        hp_br = hp_tl + rect1[3]

        # Current bounding box coordinates
        wc_tl = rect2[0]
        hc_tl = rect2[1]
        wc_br = wc_tl + rect2[2]
        hc_br = hc_tl + rect2[3]

        # Intersection
        wi_tl = max(wp_tl, wc_tl)
        hi_tl = max(hp_tl, hc_tl)
        wi_br = min(wp_br, wc_br)
        hi_br = min(hp_br, hc_br)

        # Updating the intersection
        if (wi_br - wi_tl <= 0) or (hi_br - hi_tl <= 0):

            return False, [0, 0, 0, 0]

        else:

            # If overlapping update to intersection
            intersection_coords = [wi_tl, hi_tl, wi_br-wi_tl, hi_br-hi_tl]
            return True, intersection_coords