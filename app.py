import streamlit as st
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_movies():
    with open("movies (1).pkl", "rb") as f:
        movies = pickle.load(f)

    return movies


@st.cache_data
def load_similarity():
    with open("similarity (1).pkl", "rb") as f:
        similarity = pickle.load(f)

    return similarity


movies = load_movies()
similarity = load_similarity()


# =========================================================
# HYBRID WEIGHTS
# =========================================================

CONTENT_WEIGHT = 0.80
RATING_WEIGHT = 0.15
POPULARITY_WEIGHT = 0.05


# =========================================================
# HEADER
# =========================================================

st.title("🎬 Movie Recommendation System")

st.markdown(
    """
    ### Discover your next favorite movie 🍿

    Recommendations are generated using:
    -  Content similarity
    -  Movie ratings
    -  Popularity
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(" Filters")

source_filter = st.sidebar.selectbox(
    "Movie Collection",
    [
        "All Movies",
        "TMDB",
        "Bollywood"
    ]
)


# =========================================================
# SEARCH
# =========================================================

search_query = st.text_input(
    "🔎 Search for a movie",
    placeholder="Example: Interstellar or 3 Idiots"
)


if search_query:

    # Search movies
    search_results = movies[
        movies["title"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_query.lower(),
            na=False
        )
    ]

    # Apply source filter
    if source_filter != "All Movies":
        search_results = search_results[
            search_results["source"] == source_filter
        ]

    # No results
    if search_results.empty:

        st.warning(
            f" No movies found for '{search_query}'"
        )

    else:

        st.success(
            f" Found {len(search_results)} movie(s)"
        )

        # Movie selection
        movie_options = search_results["title"].tolist()

        selected_movie = st.selectbox(
            "🎬 Select a movie",
            movie_options
        )


        # =================================================
        # MOVIE INFORMATION
        # =================================================

        selected_row = movies[
            movies["title"] == selected_movie
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🎬 Source",
                selected_row["source"]
            )

        with col2:

            rating = selected_row["vote_average"]

            if pd.isna(rating) or rating == 0:
                st.metric(
                    " Rating",
                    "N/A"
                )
            else:
                st.metric(
                    " Rating",
                    f"{rating:.1f}"
                )

        with col3:

            if pd.isna(selected_row["popularity"]):
                st.metric(
                    " Popularity",
                    "N/A"
                )
            else:
                st.metric(
                    " Popularity",
                    f"{selected_row['popularity']:.1f}"
                )


        # =================================================
        # RECOMMENDATION BUTTON
        # =================================================

        if st.button(
            " Recommend Movies",
            use_container_width=True
        ):

            # Get index
            movie_index = selected_row.name

            # Similarity scores
            similarity_scores = similarity[movie_index]

            # Copy dataframe
            results = movies.copy()

            # Add similarity
            results["similarity_score"] = similarity_scores


            # =============================================
            # HYBRID SCORE
            # =============================================

            results["hybrid_score"] = (
                CONTENT_WEIGHT
                * results["similarity_score"]
                +
                RATING_WEIGHT
                * results["rating_score"]
                +
                POPULARITY_WEIGHT
                * results["popularity_score"]
            )


            # Remove selected movie
            results = results.drop(
                index=movie_index
            )


            # Sort recommendations
            results = results.sort_values(
                "hybrid_score",
                ascending=False
            )


            # Top 10
            recommendations = results.head(10)


            # =============================================
            # DISPLAY
            # =============================================

            st.subheader(
                f" Recommendations for {selected_movie}"
            )

            st.caption(
                "Ranked using content similarity, "
                "ratings and popularity."
            )


            for rank, (_, row) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                st.markdown(
                    f"### {rank}. {row['title']}"
                )


                col1, col2, col3, col4 = st.columns(4)


                # Rating
                with col1:

                    rating = row["vote_average"]

                    if pd.isna(rating) or rating == 0:

                        st.metric(
                            " Rating",
                            "N/A"
                        )

                    else:

                        st.metric(
                            "  Rating",
                            f"{rating:.1f}"
                        )


                # Similarity
                with col2:

                    st.metric(
                        " Similarity",
                        f"{row['similarity_score']:.3f}"
                    )


                # Hybrid
                with col3:

                    st.metric(
                        " Hybrid Score",
                        f"{row['hybrid_score']:.3f}"
                    )


                # Source
                with col4:

                    st.metric(
                        " Source",
                        row["source"]
                    )


                st.divider()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    ---
    **Movie Recommendation System**
    
    Built using Python, Pandas, Scikit-learn, TF-IDF,
    Cosine Similarity and Hybrid Recommendation.
    """
)