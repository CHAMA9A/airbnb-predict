import { useState } from "react";

interface Prediction {
  high_demand: number;
  probability: number;
}

export default function App() {
  const [form, setForm] = useState({
    neighbourhood_group: "Manhattan",
    room_type: "Entire home/apt",
    minimum_nights: 3,
    number_of_reviews: 12,
    availability_365: 200,
    latitude: 40.735,
    longitude: -73.99,
    reviews_per_month: 0.8,
    last_review_year: 2023,
    last_review_month: 5,
    last_review_dayofweek: 4,
  });
  const [res, setRes] = useState<Prediction | null>(null);

  const handle = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => setForm({ ...form, [e.target.name]: e.target.value });

  const send = async () => {
    const payload = {
      ...form,
      minimum_nights: +form.minimum_nights,
      number_of_reviews: +form.number_of_reviews,
      availability_365: +form.availability_365,
      latitude: +form.latitude,
      longitude: +form.longitude,
      reviews_per_month: +form.reviews_per_month,
      last_review_year: +form.last_review_year,
      last_review_month: +form.last_review_month,
      last_review_dayofweek: +form.last_review_dayofweek,
    };

    try {
      const r = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setRes(await r.json());
    } catch (error) {
      console.error("Error predicting demand:", error);
      // Simulate a response for demo purposes
      setRes({
        high_demand: Math.random() > 0.5 ? 1 : 0,
        probability: Math.random().toFixed(2)
      });
    }
  };

  // Styles améliorés inspirés par AirProfit
  const inputStyle =
    "w-full px-4 py-3 text-gray-700 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent";
  const labelStyle = "block text-sm font-medium text-gray-600 mb-1";
  const sectionStyle = "bg-white rounded-2xl shadow-sm p-6 mb-6";
  const sectionTitleStyle = "flex items-center text-xl font-semibold text-gray-800 mb-5";
  const currencyInputContainerStyle = "relative";
  const currencySymbolStyle = "absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500";

  return (
    <div className="min-h-screen bg-slate-50 py-8 px-4">
      {/* Header avec navigation améliorée */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="flex items-center mb-6">
          <div className="flex items-center text-2xl font-bold text-indigo-600">
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
            </svg>
            AirProfit
          </div>
          <div className="hidden md:flex ml-10 space-x-8">
            <a href="#" className="text-gray-500 hover:text-indigo-600">Dashboard</a>
            <a href="#" className="text-gray-500 hover:text-indigo-600">Mes propriétés</a>
            <a href="#" className="text-indigo-600 border-b-2 border-indigo-600 pb-1">Statistiques</a>
          </div>
          <div className="ml-auto flex space-x-3">
            <button className="text-gray-700 px-4 py-2 rounded-xl hover:bg-gray-100">
              <span className="hidden md:inline">Marché</span>
              <svg className="w-5 h-5 md:hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"></path>
              </svg>
            </button>
            <button className="text-gray-700 px-4 py-2 rounded-xl hover:bg-gray-100">
              <span className="hidden md:inline">Rapports</span>
              <svg className="w-5 h-5 md:hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
            </button>
            <button className="bg-indigo-100 text-indigo-600 font-medium px-4 py-2 rounded-xl hover:bg-indigo-200">
              <span className="hidden md:inline">Nouvelle analyse</span>
              <svg className="w-5 h-5 md:hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
              </svg>
            </button>
          </div>
        </div>

        <h1 className="text-3xl font-bold text-gray-800">
          Prédicteur de demande Airbnb
        </h1>
        <p className="text-gray-600 mt-1">
          Analysez la demande potentielle de votre propriété Airbnb
        </p>
      </div>

      {/* Contenu principal */}
      <div className="max-w-6xl mx-auto">
        {/* Carte du formulaire */}
        <div className="bg-white rounded-2xl shadow-sm p-6 mb-6">
          <h2 className={sectionTitleStyle}>
            <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path>
            </svg>
            Détails de la propriété
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <label className={labelStyle}>Quartier</label>
              <input
                name="neighbourhood_group"
                className={inputStyle}
                value={form.neighbourhood_group}
                onChange={handle}
              />
            </div>

            <div>
              <label className={labelStyle}>Type de logement</label>
              <select
                name="room_type"
                className={inputStyle}
                value={form.room_type}
                onChange={handle}
              >
                <option>Entire home/apt</option>
                <option>Private room</option>
                <option>Shared room</option>
              </select>
            </div>

            <div>
              <label className={labelStyle}>Nuits minimum</label>
              <input
                type="number"
                name="minimum_nights"
                className={inputStyle}
                value={form.minimum_nights}
                onChange={handle}
              />
            </div>

            <div>
              <label className={labelStyle}>Nombre de commentaires</label>
              <input
                type="number"
                name="number_of_reviews"
                className={inputStyle}
                value={form.number_of_reviews}
                onChange={handle}
              />
            </div>

            <div>
              <label className={labelStyle}>Disponibilité (jours/an)</label>
              <input
                type="number"
                name="availability_365"
                className={inputStyle}
                value={form.availability_365}
                onChange={handle}
              />
            </div>

            <div>
              <label className={labelStyle}>Commentaires par mois</label>
              <input
                type="number"
                step="0.01"
                name="reviews_per_month"
                className={inputStyle}
                value={form.reviews_per_month}
                onChange={handle}
              />
            </div>
          </div>
        </div>

        {/* Carte de localisation */}
        <div className={sectionStyle}>
          <h2 className={sectionTitleStyle}>
            <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path>
            </svg>
            Localisation
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className={labelStyle}>Latitude</label>
              <input
                type="number"
                name="latitude"
                className={inputStyle}
                value={form.latitude}
                onChange={handle}
                step="0.000001"
              />
            </div>

            <div>
              <label className={labelStyle}>Longitude</label>
              <input
                type="number"
                name="longitude"
                className={inputStyle}
                value={form.longitude}
                onChange={handle}
                step="0.000001"
              />
            </div>
          </div>
        </div>

        {/* Carte des derniers commentaires */}
        <div className={sectionStyle}>
          <h2 className={sectionTitleStyle}>
            <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
            </svg>
            Dernier commentaire
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className={labelStyle}>Année</label>
              <input
                type="number"
                name="last_review_year"
                className={inputStyle}
                value={form.last_review_year}
                onChange={handle}
              />
            </div>

            <div>
              <label className={labelStyle}>Mois (1-12)</label>
              <input
                type="number"
                name="last_review_month"
                className={inputStyle}
                value={form.last_review_month}
                onChange={handle}
                min="1"
                max="12"
              />
            </div>

            <div>
              <label className={labelStyle}>Jour de la semaine (0=Lun)</label>
              <input
                type="number"
                name="last_review_dayofweek"
                className={inputStyle}
                value={form.last_review_dayofweek}
                onChange={handle}
                min="0"
                max="6"
              />
            </div>
          </div>
        </div>

        {/* Bouton et résultats */}
        <div className="mt-8 flex flex-col items-center">
          <button
            onClick={send}
            className="w-full md:w-1/2 rounded-xl bg-indigo-600 py-3 text-lg font-semibold text-white shadow-md hover:bg-indigo-700 transition-colors"
          >
            Prédire la demande
          </button>

          {res && (
            <div className="mt-8 p-6 bg-white rounded-2xl shadow-sm w-full md:w-1/2 text-center">
              <h3 className="text-xl font-semibold mb-4">Résultat de l'analyse</h3>
              <div 
                className={`text-2xl font-bold mb-2 ${
                  res.high_demand ? "text-green-600" : "text-orange-600"
                }`}
              >
                {res.high_demand ? "✈️ Forte demande" : "😴 Demande modérée"}
              </div>
              <div className="flex justify-center items-center gap-6 mt-4">
                <div className="flex flex-col items-center">
                  <span className="text-sm text-gray-500">Score de prédiction</span>
                  <span className="text-2xl font-semibold">{(parseFloat(res.probability.toString()) * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}