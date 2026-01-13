using IoTPlatform.Api.Services;

var builder = WebApplication.CreateBuilder(args);

// 1️⃣ CORS Politikasını Tanımla
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll",
        policy =>
        {
            policy.AllowAnyOrigin()
                  .AllowAnyMethod()
                  .AllowAnyHeader();
        });
});

// 2️⃣ Servisleri Ekle
builder.Services.AddControllers(); // Denetleyicileri (Controllers) sisteme kaydeder
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// MQTT ve AI Servislerini Kaydet
builder.Services.AddHostedService<MqttIngestService>();
builder.Services.AddHttpClient<AiService>(client =>
{
    client.BaseAddress = new Uri("http://127.0.0.1:8001");
});

var app = builder.Build();

// 3️⃣ CORS Politikasını Aktif Et
app.UseCors("AllowAll");

// Geliştirme Ortamı Ayarları
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// 4️⃣ Endpoint Yönlendirmeleri
app.MapControllers(); // Controller sınıflarındaki rotaların (Route) çalışmasını sağlar

// Sağlık Kontrolü Endpoint'i
app.MapGet("/health", () => Results.Ok(new { status = "ok", time = DateTime.Now }));

app.Run();